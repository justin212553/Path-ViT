"""
scripts/pool_multiseed_external_preds.py의 --full-train 전용 버전 — fold 축이 없다(코호트
전체를 train으로 쓰므로 k-fold 자체가 없음). train_light.py --full-train --external이 저장한
.logs/external_preds/{dataset}_{model}_FULLTRAIN_seed{seed}.csv 여러 개를 모아 (1) seed별
external c-index 분포와 (2) seed 간 예측 평균 앙상블 결과를 계산한다.

배경(2026-09-02): internal pooled/ensembled k-fold c-index가 fold당 표본이 너무 작고(N~30)
cross-fold 모델 보정 불일치까지 겹쳐 신뢰하기 어렵다는 게 확인된 뒤(findings_backlog.md),
"TCGA 전체를 train으로 쓰고 고정 epoch만 돌린 뒤 external만, 시드만 여러 개 반복해서 평균±CI로
비교" 프로토콜(사용자 제안)의 결과 취합용.

사용법:
    python scripts/pool_fulltrain_external_preds.py --dataset cptac --model M7_PW8_STG_R_COX_ADD \
        --seeds 42,84,126,168,210 --bootstrap 2000
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.metrics import compute_survival_metrics


def _load_seed_predictions(pred_dir: Path, dataset: str, model: str, seed: int) -> dict:
    path = pred_dir / f"{dataset}_{model}_FULLTRAIN_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"seed={seed} external 예측 파일을 못 찾음: {path}")
    preds = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            preds[row["case_id"]] = (float(row["risk"]), float(row["OS_time"]), int(row["OS_event"]))
    return preds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    pred_dir = Path(__file__).parent.parent / ".logs" / "external_preds"

    per_seed_preds = {}
    per_seed_c = []
    print(f"=== --full-train 반복시드 external — {args.dataset} {args.model} ===")
    for seed in seeds:
        preds = _load_seed_predictions(pred_dir, args.dataset, args.model, seed)
        per_seed_preds[seed] = preds
        case_ids = list(preds.keys())
        risks = np.array([preds[c][0] for c in case_ids])
        times = np.array([preds[c][1] for c in case_ids])
        events = np.array([preds[c][2] for c in case_ids])
        m = compute_survival_metrics(risks, times, events)
        per_seed_c.append(m["c_index"])
        print(f"  seed={seed}: N={len(case_ids)} | c_index={m['c_index']:.4f} | HR={m['hr']:.3f} "
              f"[{m['hr_ci_lower']:.3f}, {m['hr_ci_upper']:.3f}] | log_rank_p={m['log_rank_p']:.4f}")

    per_seed_c = np.array(per_seed_c)
    print(f"  -> seed 간: mean={per_seed_c.mean():.4f}, std={per_seed_c.std():.4f}, "
          f"min={per_seed_c.min():.4f}, max={per_seed_c.max():.4f}")

    common_cases = sorted(set.intersection(*(set(p.keys()) for p in per_seed_preds.values())))
    ensembled_risks, times, events = [], [], []
    for cid in common_cases:
        seed_risks, ref_time, ref_event = [], None, None
        for seed in seeds:
            r, t, e = per_seed_preds[seed][cid]
            seed_risks.append(r)
            if ref_time is None:
                ref_time, ref_event = t, e
        ensembled_risks.append(float(np.mean(seed_risks)))
        times.append(ref_time)
        events.append(ref_event)
    ensembled_risks, times, events = np.array(ensembled_risks), np.array(times), np.array(events)
    m = compute_survival_metrics(ensembled_risks, times, events)
    print(f"\n=== seed 간 예측 평균 앙상블 (N={len(common_cases)}, events={int(events.sum())}) ===")
    print(f"  c_index={m['c_index']:.4f} | HR={m['hr']:.3f} [{m['hr_ci_lower']:.3f}, {m['hr_ci_upper']:.3f}] | "
          f"log_rank_p={m['log_rank_p']:.4f}")

    if args.bootstrap > 0:
        rng = np.random.RandomState(0)
        n = len(common_cases)
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
