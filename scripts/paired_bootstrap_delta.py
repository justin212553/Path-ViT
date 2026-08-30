"""
두 모델의 C-index를 paired bootstrap으로 직접 비교한다 — 두 모델을 따로 resample해서 CI가
겹치는지 보는(scripts/pool_multiseed_kfold_preds.py, pool_multiseed_external_preds.py의 방식)
대신, 매 bootstrap 회차마다 "같은 환자 집합"을 resample해서 그 안에서 두 모델의 C-index를 함께
계산하고 delta(=model_b − model_a)를 기록한다. 같은 환자에 대한 두 모델의 예측이 짝지어져
있다는 걸 활용해 분산을 줄이는(independent 비교보다 검정력이 높은) 방법 — 2026-08-21, 외부
피드백(paired bootstrap on delta 제안)을 반영해 추가.

환자 단위 risk score 앙상블 구성은 기존 pooling 스크립트와 완전히 동일한 규칙을 그대로
재구현한다(따로 재학습/재추론 없이 이미 저장된 예측 CSV만 사용):
  - internal(--split internal): pool_multiseed_kfold_preds.py와 동일 — seed마다 그 환자를 한
    번도 학습에 안 쓴 5-fold OOF 예측을 얻고, 환자 단위로 seed 간 평균.
  - external(--split external): pool_multiseed_external_preds.py와 동일 — seed x fold
    체크포인트 전부(환자를 학습에 쓴 적이 없으므로 제약 없음)의 예측을 환자 단위로 평균.

두 모델의 환자 집합이 다를 수 있어(예: WSI 계열과 clinic/RNA-only 계열이 다른 case pool을
쓸 가능성) 반드시 교집합만 사용하고, 교집합이 아닌 환자가 있으면 경고한다.

bootstrap resample은 매 회차 동일한 index array를 두 모델에 똑같이 적용(핵심 - 이게 "paired"의
정의)하고, np.random.RandomState(0)으로 고정해 재현 가능하게 한다(기존 pooling 스크립트의
단일모델 bootstrap과 같은 seed=0 관례).

p-value = 2 * min(P(delta<=0), P(delta>=0)) — 양측검정, delta 분포가 완전히 한쪽으로 쏠리면
자동으로 <=1로 clip됨(수학적으로 항상 <=1).

사용법:
    python scripts/paired_bootstrap_delta.py --split internal --dataset tcga \
        --model-a M1_POOL_uni2native_SS_DISP \
        --model-b PMA_uni2native_INT1500_SS_AUX_NOCLINICAL_DISP \
        --seeds 84,126 --n-folds 5 --bootstrap 2000

    python scripts/paired_bootstrap_delta.py --split external --dataset cptac \
        --model-a M1_POOL_uni2native_SS_DISP \
        --model-b PMA_uni2native_INT1500_SS_AUX_NOCLINICAL_DISP \
        --seeds 84,126 --n-folds 5 --bootstrap 2000
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


def _fast_c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """utils.metrics.compute_survival_metrics의 c_index 계산부만 떼어낸 버전 — HR/log-rank
    (lifelines CoxPHFitter.fit, 매 호출마다 수렴 반복이 있어 느림)를 건너뛴다. bootstrap
    루프에서 c_index만 필요할 때 이걸 쓰면 4000회(2모델 x 2000회) 호출도 수 초면 끝난다."""
    risk = np.asarray(risk, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    comparable = (time[:, None] < time[None, :]) & event[:, None]
    n_permissible = int(comparable.sum())
    if n_permissible == 0:
        return float("nan")
    concordant = comparable & (risk[:, None] > risk[None, :])
    tied_risk = comparable & (risk[:, None] == risk[None, :])
    return float((concordant.sum() + 0.5 * tied_risk.sum()) / n_permissible)


def _load_seed_predictions(pred_dir: Path, dataset: str, model: str, seed: int, n_folds: int) -> dict:
    """internal 전용 — pool_multiseed_kfold_preds.py와 동일."""
    preds = {}
    for fold in range(n_folds):
        path = pred_dir / f"{dataset}_{model}_FOLD{fold}OF{n_folds}_seed{seed}_fold{fold}of{n_folds}.csv"
        if not path.exists():
            path = pred_dir / f"{dataset}_{model}_seed{seed}_fold{fold}of{n_folds}.csv"
        if not path.exists():
            raise FileNotFoundError(f"seed={seed} fold={fold} 예측 파일을 못 찾음: {path}")
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cid = row["case_id"]
                if cid in preds:
                    raise ValueError(f"seed={seed} 내에서 case_id 중복: {cid}")
                preds[cid] = (float(row["risk"]), float(row["OS_time"]), int(row["OS_event"]))
    return preds


def _load_run_predictions(pred_dir: Path, dataset: str, model: str, seed: int, fold: int, n_folds: int) -> dict:
    """external 전용 — pool_multiseed_external_preds.py와 동일."""
    path = pred_dir / f"{dataset}_{model}_FOLD{fold}OF{n_folds}_seed{seed}_fold{fold}of{n_folds}.csv"
    if not path.exists():
        path = pred_dir / f"{dataset}_{model}_seed{seed}_fold{fold}of{n_folds}.csv"
    if not path.exists():
        raise FileNotFoundError(f"seed={seed} fold={fold} external 예측 파일을 못 찾음: {path}")
    preds = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            preds[row["case_id"]] = (float(row["risk"]), float(row["OS_time"]), int(row["OS_event"]))
    return preds


def _ensemble_internal(pred_dir: Path, dataset: str, model: str, seeds: list[int], n_folds: int):
    per_seed_preds = {seed: _load_seed_predictions(pred_dir, dataset, model, seed, n_folds) for seed in seeds}
    case_sets = [set(p.keys()) for p in per_seed_preds.values()]
    common = sorted(set.intersection(*case_sets))
    risks, times, events = [], [], []
    for cid in common:
        seed_risks = []
        ref_time = ref_event = None
        for seed in seeds:
            r, t, e = per_seed_preds[seed][cid]
            seed_risks.append(r)
            if ref_time is None:
                ref_time, ref_event = t, e
        risks.append(float(np.mean(seed_risks)))
        times.append(ref_time)
        events.append(ref_event)
    return common, np.array(risks), np.array(times), np.array(events)


def _ensemble_external(pred_dir: Path, dataset: str, model: str, seeds: list[int], n_folds: int):
    patient_risks: dict[str, list[float]] = defaultdict(list)
    patient_label: dict[str, tuple[float, int]] = {}
    for seed in seeds:
        for fold in range(n_folds):
            preds = _load_run_predictions(pred_dir, dataset, model, seed, fold, n_folds)
            for cid, (r, t, e) in preds.items():
                patient_risks[cid].append(r)
                patient_label[cid] = (t, e)
    case_ids = sorted(patient_risks.keys())
    risks = np.array([float(np.mean(patient_risks[c])) for c in case_ids])
    times = np.array([patient_label[c][0] for c in case_ids])
    events = np.array([patient_label[c][1] for c in case_ids])
    return case_ids, risks, times, events


def paired_bootstrap_delta(
    case_ids_a, risks_a, times_a, events_a,
    case_ids_b, risks_b, times_b, events_b,
    n_bootstrap: int = 2000, seed: int = 0,
) -> dict:
    """model_b − model_a의 paired bootstrap delta 분포와 point estimate를 계산."""
    idx_a = {c: i for i, c in enumerate(case_ids_a)}
    idx_b = {c: i for i, c in enumerate(case_ids_b)}
    common = sorted(set(case_ids_a) & set(case_ids_b))
    if len(common) < len(case_ids_a) or len(common) < len(case_ids_b):
        only_a = set(case_ids_a) - set(common)
        only_b = set(case_ids_b) - set(common)
        print(f"  [경고] 두 모델의 환자 집합이 다름 — A만 있음 {len(only_a)}명, B만 있음 {len(only_b)}명. "
              f"교집합 {len(common)}명만 paired 비교에 사용.")

    ia = np.array([idx_a[c] for c in common])
    ib = np.array([idx_b[c] for c in common])
    ra, ta, ea = risks_a[ia], times_a[ia], events_a[ia]
    rb, tb, eb = risks_b[ib], times_b[ib], events_b[ib]
    # 라벨(OS_time/OS_event) 불일치는 같은 환자인데 다른 코호트 파이프라인을 거쳤을 가능성을
    # 시사하므로 명시적으로 검증한다.
    mismatched = np.sum((np.abs(ta - tb) > 1e-6) | (ea != eb))
    if mismatched > 0:
        print(f"  [경고] {mismatched}명은 OS_time/OS_event가 두 모델 파일 간에 다름 — 라벨 불일치 의심.")

    n = len(common)
    point_a = compute_survival_metrics(ra, ta, ea)["c_index"]
    point_b = compute_survival_metrics(rb, tb, eb)["c_index"]

    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_bootstrap):
        boot_idx = rng.randint(0, n, n)
        ca = _fast_c_index(ra[boot_idx], ta[boot_idx], ea[boot_idx])
        cb = _fast_c_index(rb[boot_idx], tb[boot_idx], eb[boot_idx])
        if np.isnan(ca) or np.isnan(cb):
            continue
        deltas.append(cb - ca)
    deltas = np.array(deltas)

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_le0 = float(np.mean(deltas <= 0))
    p_ge0 = float(np.mean(deltas >= 0))
    p_value = min(2 * min(p_le0, p_ge0), 1.0)

    return {
        "n_patients": n,
        "point_a": point_a,
        "point_b": point_b,
        "point_delta": point_b - point_a,
        "n_boot_used": len(deltas),
        "delta_mean": float(deltas.mean()),
        "delta_ci_lo": float(lo),
        "delta_ci_hi": float(hi),
        "p_value": p_value,
        "significant": not (lo <= 0 <= hi),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=str, required=True, choices=["internal", "external"])
    parser.add_argument("--dataset", type=str, required=True, choices=["tcga", "cptac", "both"],
                         help="internal이면 학습 코호트(예: tcga), external이면 평가 대상 코호트(예: cptac).")
    parser.add_argument("--model-a", type=str, required=True, help="model_prefix (baseline, delta의 기준)")
    parser.add_argument("--model-b", type=str, required=True, help="model_prefix (비교 대상, delta = B - A)")
    parser.add_argument("--seeds", type=str, default="84,126")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--pred-root", type=str, default=None,
                         help="예측 CSV가 있는 루트 디렉터리(기본: .logs, "
                              "paper/final_preds_snapshot 스냅샷을 쓰려면 이 값으로 지정)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    pred_root = Path(args.pred_root) if args.pred_root else _ROOT / ".logs"
    if args.split == "internal":
        pred_dir = pred_root / "kfold_preds"
        ensemble_fn = _ensemble_internal
    else:
        pred_dir = pred_root / "external_preds"
        ensemble_fn = _ensemble_external

    cases_a, risks_a, times_a, events_a = ensemble_fn(pred_dir, args.dataset, args.model_a, seeds, args.n_folds)
    cases_b, risks_b, times_b, events_b = ensemble_fn(pred_dir, args.dataset, args.model_b, seeds, args.n_folds)

    print(f"=== paired bootstrap delta ({args.split}, {args.dataset}) ===")
    print(f"  A = {args.model_a} (N={len(cases_a)})")
    print(f"  B = {args.model_b} (N={len(cases_b)})")

    result = paired_bootstrap_delta(cases_a, risks_a, times_a, events_a,
                                     cases_b, risks_b, times_b, events_b,
                                     n_bootstrap=args.bootstrap)
    sig = "유의(CI가 0 안 포함)" if result["significant"] else "비유의(CI가 0 포함)"
    print(f"  N(paired)={result['n_patients']} | C(A)={result['point_a']:.4f} | C(B)={result['point_b']:.4f} | "
          f"point delta(B-A)={result['point_delta']:+.4f}")
    print(f"  bootstrap(n={result['n_boot_used']}) delta 95% CI = "
          f"[{result['delta_ci_lo']:+.4f}, {result['delta_ci_hi']:+.4f}] (mean={result['delta_mean']:+.4f})")
    print(f"  two-sided p = {result['p_value']:.4f} -> {sig}")


if __name__ == "__main__":
    main()
