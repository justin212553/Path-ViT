"""
scripts/paired_bootstrap_delta.py의 --full-train 전용 버전 — fold 축이 없다. train.py/
train_light.py --full-train --external이 저장한 .logs/external_preds/{dataset}_{model}_
FULLTRAIN_seed{seed}.csv 여러 개(시드 반복)를 모델 A/B 각각 seed 평균으로 앙상블한 뒤,
같은 환자 집합에 paired bootstrap(scripts/paired_bootstrap_delta.py::paired_bootstrap_delta,
그대로 재사용)으로 delta(B−A) 유의성을 계산한다.

주 용도(2026-09-02): "RNA+Clinical만(M7) vs WSI 추가(PMA/HDP_Pretrain_Cluster)"처럼 모달리티
추가가 external에서 통계적으로 유의한 순증분인지 확인 — internal k-fold pooled c-index의
신뢰성 문제(findings_backlog.md) 때문에 이 프로토콜(전체 train+고정epoch+external만+다시드)로
비교하기로 함.

사용법:
    python scripts/paired_bootstrap_delta_fulltrain.py --dataset cptac \
        --model-a M7_PW8_STG_R_COX_ADD --model-b PMA_uni2_PW8_SS_AUX_STG_R_DISP_COX_ADD \
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

from scripts.paired_bootstrap_delta import paired_bootstrap_delta


def _ensemble_fulltrain(pred_dir: Path, dataset: str, model: str, seeds: list[int]):
    per_seed_preds = {}
    for seed in seeds:
        path = pred_dir / f"{dataset}_{model}_FULLTRAIN_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(f"seed={seed} 예측 파일을 못 찾음: {path}")
        preds = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                preds[row["case_id"]] = (float(row["risk"]), float(row["OS_time"]), int(row["OS_event"]))
        per_seed_preds[seed] = preds

    case_sets = [set(p.keys()) for p in per_seed_preds.values()]
    common = sorted(set.intersection(*case_sets))
    risks, times, events = [], [], []
    for cid in common:
        seed_risks, ref_time, ref_event = [], None, None
        for seed in seeds:
            r, t, e = per_seed_preds[seed][cid]
            seed_risks.append(r)
            if ref_time is None:
                ref_time, ref_event = t, e
        risks.append(float(np.mean(seed_risks)))
        times.append(ref_time)
        events.append(ref_event)
    return common, np.array(risks), np.array(times), np.array(events)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True, help="external 코호트(예: cptac)")
    parser.add_argument("--model-a", type=str, required=True, help="model_prefix (baseline, delta의 기준)")
    parser.add_argument("--model-b", type=str, required=True, help="model_prefix (비교 대상, delta = B - A)")
    parser.add_argument("--seeds", type=str, default="42,84,126,168,210")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    pred_dir = _ROOT / ".logs" / "external_preds"

    cases_a, risks_a, times_a, events_a = _ensemble_fulltrain(pred_dir, args.dataset, args.model_a, seeds)
    cases_b, risks_b, times_b, events_b = _ensemble_fulltrain(pred_dir, args.dataset, args.model_b, seeds)

    print(f"=== paired bootstrap delta (--full-train, {args.dataset}) ===")
    print(f"  A = {args.model_a} (N={len(cases_a)})")
    print(f"  B = {args.model_b} (N={len(cases_b)})")

    result = paired_bootstrap_delta(cases_a, risks_a, times_a, events_a,
                                     cases_b, risks_b, times_b, events_b,
                                     n_bootstrap=args.bootstrap)
    sig = "유의(CI가 0 안 포함)" if result["significant"] else "비유의(CI가 0 포함)"
    print(f"  N(paired)={result['n_patients']} | C(A)={result['point_a']:.4f} | C(B)={result['point_b']:.4f} | "
          f"point_delta={result['point_delta']:+.4f}")
    print(f"  bootstrap delta: mean={result['delta_mean']:+.4f}, 95% CI=[{result['delta_ci_lo']:+.4f}, "
          f"{result['delta_ci_hi']:+.4f}], p={result['p_value']:.4f} -> {sig}")


if __name__ == "__main__":
    main()
