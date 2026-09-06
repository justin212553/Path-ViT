"""
external 코호트(반대 기관 전체)를 여러 (seed, fold) checkpoint로 각각 평가한 예측을 모아,
(1) 실행별(seed x fold) external 지표 분포와 (2) 환자 단위 예측 평균 앙상블 결과를 함께 계산한다.

train.py --eval-external-ckpt(2026-08-08 추가)로 뽑은 .logs/external_preds/*.csv를 입력으로
쓴다. scripts/pool_multiseed_kfold_preds.py(internal용)와 달리 seed 간 "겹치지 않는 held-out
환자만 골라 평균"할 필요가 없다 — external 코호트는 어느 (seed, fold) checkpoint로 평가해도
"그 환자를 학습에 쓴 적이 전혀 없다"는 전제가 항상 성립하기 때문이다(코호트 자체가 학습
데이터에서 완전히 배제됨, --dataset tcga --external이면 cptac은 어떤 seed/fold 조합에서도
학습에 등장하지 않음). 그래서 3seed x 5fold=15개 checkpoint의 예측 전부를 아무 제약 없이
환자 단위로 그대로 평균해도 된다.

사용법:
    python scripts/pool_multiseed_external_preds.py --dataset cptac \
        --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.metrics import compute_survival_metrics


def _load_run_predictions(pred_dir: Path, dataset: str, model: str, seed: int, fold: int, n_folds: int) -> dict:
    # 2026-09-06: model_prefix를 손으로 완벽히 재현해야 하는 두 개 하드코딩 패턴(with/without
    # "_FOLD{fold}OF{n_folds}") 대신 글롭으로 찾는다 — scripts/pool_multiseed_kfold_preds.py::
    # _find_pred_path와 동일한 이유/관례.
    suffix = f"_seed{seed}_fold{fold}of{n_folds}.csv"
    matches = sorted(pred_dir.glob(f"{dataset}_{model}*{suffix}"))
    if len(matches) > 1:
        raise ValueError(
            f"seed={seed} fold={fold}: '{dataset}_{model}*{suffix}' 패턴에 여러 파일이 걸림 — "
            f"{[p.name for p in matches]}"
        )
    if not matches:
        nearby = sorted(pred_dir.glob(f"{dataset}_*{suffix}"))
        hint = f" (같은 seed/fold의 다른 태그 후보: {[p.name for p in nearby]})" if nearby else " (같은 seed/fold 파일 자체가 없음 — 그 실행이 아직 안 끝났거나 실패했을 가능성)"
        raise FileNotFoundError(f"seed={seed} fold={fold} external 예측 파일을 못 찾음: {pred_dir}/{dataset}_{model}*{suffix}{hint}")
    path = matches[0]
    preds = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            preds[row["case_id"]] = (float(row["risk"]), float(row["OS_time"]), int(row["OS_event"]))
    return preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["tcga", "cptac", "both", "brca"],
        help="external 코호트(학습에 쓰인 반대쪽) — 예: --dataset tcga --external로 학습했으면 "
             "여기엔 cptac을 준다(train.py의 external_dataset 파일명 규약과 동일). "
             "2026-09-01: 'brca'는 scripts/train_brca_m4.py/train_brca_m7.py --fold의 "
             "institution(BH) holdout 결과용(파일명 접두사가 brca_라 이 값으로 고정).",
    )
    parser.add_argument("--model", type=str, required=True,
                         help="model_prefix (예: PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD)")
    parser.add_argument("--seeds", type=str, default="42,84,126",
                         help="콤마로 구분한 seed 목록 (기본: 42,84,126 — 이 프로젝트 표준 3시드)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--bootstrap", type=int, default=0,
        help="주어지면(예: 2000) 최종 앙상블 지표에 환자 단위 resample bootstrap 95%% CI를 추가로 "
             "계산한다. 0(기본)이면 생략.",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    pred_dir = Path(__file__).parent.parent / ".logs" / "external_preds"

    run_c = []
    patient_risks: dict[str, list[float]] = defaultdict(list)
    patient_label: dict[str, tuple[float, int]] = {}

    print(f"=== 실행별(seed x fold) external — {args.dataset} {args.model} ===")
    for seed in seeds:
        for fold in range(args.n_folds):
            preds = _load_run_predictions(pred_dir, args.dataset, args.model, seed, fold, args.n_folds)
            case_ids = list(preds.keys())
            risks  = np.array([preds[c][0] for c in case_ids])
            times  = np.array([preds[c][1] for c in case_ids])
            events = np.array([preds[c][2] for c in case_ids])
            m = compute_survival_metrics(risks, times, events)
            run_c.append(m["c_index"])
            print(f"  seed={seed} fold={fold}: N={len(case_ids)}, events={int(events.sum())} | "
                  f"c_index={m['c_index']:.4f} | HR={m['hr']:.3f} [{m['hr_ci_lower']:.3f}, {m['hr_ci_upper']:.3f}] | "
                  f"log_rank_p={m['log_rank_p']:.4f}")
            for cid, (r, t, e) in preds.items():
                patient_risks[cid].append(r)
                if cid in patient_label and patient_label[cid] != (t, e):
                    raise ValueError(f"case_id={cid}의 OS_time/OS_event가 실행마다 다름 — 라벨 불일치 의심")
                patient_label[cid] = (t, e)

    run_c = np.array(run_c)
    print(f"  -> {len(run_c)}개 실행(seed x fold) 간 external c-index: "
          f"mean={run_c.mean():.4f}, std={run_c.std():.4f}")

    case_ids = sorted(patient_risks.keys())
    n_runs_per_patient = {len(patient_risks[c]) for c in case_ids}
    if len(n_runs_per_patient) > 1:
        print(f"  [경고] 환자별로 모인 예측 개수가 다릅니다({n_runs_per_patient}) — "
              f"일부 (seed,fold) CSV가 누락됐을 수 있습니다.")

    ensembled_risks = np.array([float(np.mean(patient_risks[c])) for c in case_ids])
    times  = np.array([patient_label[c][0] for c in case_ids])
    events = np.array([patient_label[c][1] for c in case_ids])
    m = compute_survival_metrics(ensembled_risks, times, events)
    print(f"\n=== {len(seeds)}seed x {args.n_folds}fold 전체({len(run_c)}개) 예측 평균 앙상블 "
          f"(N={len(case_ids)}, events={int(events.sum())}) ===")
    print(f"  c_index={m['c_index']:.4f} | HR={m['hr']:.3f} [{m['hr_ci_lower']:.3f}, {m['hr_ci_upper']:.3f}] | "
          f"log_rank_p={m['log_rank_p']:.4f}")

    if args.bootstrap > 0:
        rng = np.random.RandomState(0)
        n = len(case_ids)
        boot_c = []
        for _ in range(args.bootstrap):
            idx = rng.randint(0, n, n)
            bm = compute_survival_metrics(ensembled_risks[idx], times[idx], events[idx])
            if not np.isnan(bm["c_index"]):
                boot_c.append(bm["c_index"])
        boot_c = np.array(boot_c)
        lo, hi = np.percentile(boot_c, [2.5, 97.5])
        print(f"  bootstrap 95% CI (환자 단위 resample, n={args.bootstrap}) = "
              f"[{lo:.4f}, {hi:.4f}], std={boot_c.std():.4f}")


if __name__ == "__main__":
    main()
