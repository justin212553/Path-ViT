"""
K-fold(--fold, train.py/train_light.py) 학습 결과를 internal/external 둘 다 한 번에 집계한다.

internal과 external은 집계 방식이 다르다는 점이 이 스크립트의 핵심 — 둘을 섞어 계산하면 안 된다.

- internal: fold마다 서로 겹치지 않는 코호트의 20%를 test로 평가하므로(pool_kfold_preds.py
  참조), n_folds개를 이어붙이면 코호트 전체 크기의 risk score 집합이 된다 — "풀링(pooling)"
  해서 그 전체로 c-index를 한 번 계산한다.
- external: 모든 fold가 "같은" 반대 코호트 전체를 평가하지만, 그 평가에 쓰는 모델이 fold마다
  다르다(각 fold는 서로 다른 train 부분집합으로 학습됨) — pooling할 대상(서로 다른 환자)이
  아니라 "같은 환자 집합에 대한 5개의 서로 다른 추정치"이므로, fold별 external c-index를
  단순 평균(및 표준편차)해서 본다.

internal은 .logs/kfold_preds/*.csv(train.py/train_light.py --fold가 저장)를 읽고,
external은 .logs/train_{dataset}_seed{seed}_{model}_kfold5_fold{N}.log(scripts/train_m*_kfold_hpc.sh,
_pma_ex_ss_aux_tileaugment_dispersion_kfold_array.sh가 쓰는 로그 파일명 관례)에서
"external_c_index=..." 줄을 정규식으로 뽑는다 — "final_external_c_index="는 별도 코드 경로라
제외한다(train.py에 --swa 등 조건부로 두 번 찍힐 수 있음, 확인된 값이 다를 수 있어 대표값으로
쓰지 않음).

사용법:
    python scripts/summarize_kfold.py --dataset tcga --seed 42 --n-folds 5 --model M1_SS_AUG_DISP
    python scripts/summarize_kfold.py --dataset tcga --seed 42 --n-folds 5 --model PMA_EX_SS_AUX_AUG_DISP
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.metrics import compute_survival_metrics

# "final_external_c_index=..."는 매치하지 않도록 앞에 "final_"이 없는 경우만 잡는다.
_EXTERNAL_RE = re.compile(
    r"(?<!final_)external_c_index=([\d.]+) \| external_HR=([\d.]+) .*? \| "
    r"external_logrank_p=([\d.]+)"
)


def _pool_internal(dataset: str, model: str, seed: int, n_folds: int) -> dict:
    pred_dir = Path(__file__).parent.parent / ".logs" / "kfold_preds"
    case_ids, risks, times, events = [], [], [], []
    for fold in range(n_folds):
        path = pred_dir / f"{dataset}_{model}_FOLD{fold}OF{n_folds}_seed{seed}_fold{fold}of{n_folds}.csv"
        if not path.exists():
            path = pred_dir / f"{dataset}_{model}_seed{seed}_fold{fold}of{n_folds}.csv"
        if not path.exists():
            raise FileNotFoundError(f"fold {fold} internal 예측 파일을 못 찾음: {path}")
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                case_ids.append(row["case_id"])
                risks.append(float(row["risk"]))
                times.append(float(row["OS_time"]))
                events.append(int(row["OS_event"]))

    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"pooled case_id에 중복이 있습니다({len(case_ids)}개 중 {len(set(case_ids))}개 유일) "
                          "— fold 분할이 겹쳤을 가능성이 있습니다.")

    risks, times, events = np.array(risks), np.array(times), np.array(events)
    metrics = compute_survival_metrics(risks, times, events)
    return {"n": len(case_ids), "events": int(events.sum()), **metrics}


def _average_external(dataset: str, model: str, seed: int, n_folds: int) -> dict:
    log_dir = Path(__file__).parent.parent / ".logs"
    c_indices, hrs, ps = [], [], []
    missing = []
    for fold in range(n_folds):
        log_path = log_dir / f"train_{dataset}_seed{seed}_{model}_kfold5_fold{fold}.log"
        if not log_path.exists():
            missing.append(log_path)
            continue
        text = log_path.read_text(errors="ignore")
        matches = _EXTERNAL_RE.findall(text)
        if not matches:
            missing.append(log_path)
            continue
        c, hr, p = matches[-1]  # 로그에 여러 번 찍혀도 마지막(최종) 값을 쓴다
        c_indices.append(float(c))
        hrs.append(float(hr))
        ps.append(float(p))
    if missing:
        print(f"  경고: external 값을 못 찾은 fold 로그 {len(missing)}개 (건너뜀): "
              f"{[str(p) for p in missing]}")
    return {
        "n_folds_found": len(c_indices),
        "c_index_mean": float(np.mean(c_indices)) if c_indices else float("nan"),
        "c_index_std": float(np.std(c_indices)) if c_indices else float("nan"),
        "c_indices": c_indices,
        "hr_mean": float(np.mean(hrs)) if hrs else float("nan"),
        "p_mean": float(np.mean(ps)) if ps else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True, choices=["tcga", "cptac", "both"])
    parser.add_argument("--model", type=str, required=True,
                         help="model_prefix(fold 접미사 제외) — 예: M1_SS_AUG_DISP, "
                              "PMA_EX_SS_AUX_AUG_DISP, PMA_EX_SS_AUX_AUG_NOCLINICAL_DISP")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    print(f"=== {args.dataset} {args.model} seed={args.seed} ({args.n_folds}-fold) ===\n")

    internal = _pool_internal(args.dataset, args.model, args.seed, args.n_folds)
    print(f"[internal, pooled out-of-fold] N={internal['n']} events={internal['events']}")
    print(f"  c_index={internal['c_index']:.4f} | HR={internal['hr']:.3f} "
          f"[{internal['hr_ci_lower']:.3f}, {internal['hr_ci_upper']:.3f}] | "
          f"log_rank_p={internal['log_rank_p']:.4f}\n")

    external = _average_external(args.dataset, args.model, args.seed, args.n_folds)
    print(f"[external, {external['n_folds_found']}-fold 평균 — pooling 아님, 서로 다른 모델의 "
          f"같은 코호트 추정치를 평균]")
    print(f"  c_index={external['c_index_mean']:.4f} (±{external['c_index_std']:.4f}) | "
          f"HR(평균)={external['hr_mean']:.3f} | log_rank_p(평균)={external['p_mean']:.4f}")
    print(f"  fold별 c_index: {[f'{c:.4f}' for c in external['c_indices']]}")


if __name__ == "__main__":
    main()
