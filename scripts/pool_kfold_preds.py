"""
train_light.py --fold로 각 fold마다 저장한 .logs/kfold_preds/*.csv를 모아, out-of-fold
risk score를 코호트 전체에 대해 풀링(pool)한 뒤 c-index/HR/log-rank p를 한 번에 계산한다.

각 fold가 자기 test fold의 환자를 정확히 한 번씩 평가하므로, n_folds개를 전부 모으면
코호트 전체 크기의 "internal" 평가 표본이 만들어진다(단일 6:2:2 split의 20%보다 훨씬 큼).

사용법:
    python scripts/pool_kfold_preds.py --dataset tcga --model M7_EX --seed 42 --n-folds 5
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True, choices=["tcga", "cptac", "both"])
    parser.add_argument("--model", type=str, required=True, help="model_prefix (예: M7_EX)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    pred_dir = Path(__file__).parent.parent / ".logs" / "kfold_preds"
    case_ids, risks, times, events = [], [], [], []
    for fold in range(args.n_folds):
        path = pred_dir / f"{args.dataset}_{args.model}_FOLD{fold}OF{args.n_folds}_seed{args.seed}_fold{fold}of{args.n_folds}.csv"
        if not path.exists():
            # model_prefix 자체에 _FOLD{f}OF{n} 접미사가 이미 붙어 저장되므로 접미사 없는 이름도 시도
            path = pred_dir / f"{args.dataset}_{args.model}_seed{args.seed}_fold{fold}of{args.n_folds}.csv"
        if not path.exists():
            raise FileNotFoundError(f"fold {fold} 예측 파일을 못 찾음: {path}")
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_ids.append(row["case_id"])
                risks.append(float(row["risk"]))
                times.append(float(row["OS_time"]))
                events.append(int(row["OS_event"]))

    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"pooled case_id에 중복이 있습니다({len(case_ids)}개 중 {len(set(case_ids))}개 유일) "
                          "— fold 분할이 겹쳤을 가능성이 있습니다.")

    risks, times, events = np.array(risks), np.array(times), np.array(events)
    metrics = compute_survival_metrics(risks, times, events)
    print(f"Pooled out-of-fold internal test — {args.dataset} {args.model} seed={args.seed} "
          f"({args.n_folds}-fold, N={len(case_ids)}, events={int(events.sum())})")
    print(f"  c_index={metrics['c_index']:.4f} | HR={metrics['hr']:.3f} "
          f"[{metrics['hr_ci_lower']:.3f}, {metrics['hr_ci_upper']:.3f}] | log_rank_p={metrics['log_rank_p']:.4f}")


if __name__ == "__main__":
    main()
