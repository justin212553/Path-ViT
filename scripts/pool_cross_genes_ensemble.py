"""
서로 다른 RNA 유전자셋으로 각각 학습된 모델들을 환자 단위로 예측 앙상블한다(seed 간 앙상블과
같은 원리를 "유전자셋"이라는 축으로 확장) — 2026-09-06 사용자 제안: "유전자셋을 하나로
확정하려 하지 말고, 서로 다른 유전자셋으로 배운 모델들끼리 앙상블시켜보자". 세 번의 "유전자
수를 1500 이상으로 늘리는" 시도가 전부 external 유의하게 악화(코호트 라벨 사용 여부와 무관하게
전부 같은 방향)된 뒤 — 유전자셋 자체를 하나로 더 키우는 대신, 서로 다른(각자 leak-free) 유전자셋
모델의 예측을 사후에 평균 내는 쪽이 "한 모델에 다 욱여넣기"보다 나을 수 있다는 가설.

기존 학습을 재사용한다(새 학습 불필요) — scripts/pool_multiseed_kfold_preds.py(internal)/
pool_multiseed_external_preds.py(external)와 동일한 예측 CSV(.logs/kfold_preds,
.logs/external_preds)를 그대로 읽는다. 여러 --model 태그 x 여러 --seeds(internal은 seed만,
external은 seed x fold)의 risk score를 전부 환자 단위로 평균한다.

**주의**: internal의 "held-out" 원칙(그 환자를 학습에 한 번도 안 쓴 모델의 예측만 평균)은 유전자셋이
달라도 유지된다 — 각 (model, seed) 조합이 그 환자가 val fold였던 checkpoint의 예측만 제공하므로.

사용법:
    python scripts/pool_cross_genes_ensemble.py --dataset tcga --split internal \
        --models PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1,PORPOISE_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
        --seeds 84,126,168,210,252 --n-folds 5 --bootstrap 2000
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.metrics import compute_survival_metrics

# pool_multiseed_kfold_preds.py::_SPECIAL_INFIXES/글롭 접두사-숫자 모호성 필터와 동일 —
# 여기서만 쓰는 작은 로더라 중복 정의하되, 로직은 두 원본 스크립트와 동일하게 맞춘다.
_SPECIAL_INFIXES = ("FINALEPOCH", "SOUP", "FULLTRAIN")


def _find_pred_path(pred_dir: Path, dataset: str, model: str, suffix: str) -> Path:
    prefix = f"{dataset}_{model}"
    matches = sorted(pred_dir.glob(f"{prefix}*{suffix}"))
    matches = [
        p for p in matches
        if not p.name[len(prefix):len(prefix) + 1].isdigit()
        and all(tag in model or tag not in p.name for tag in _SPECIAL_INFIXES)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"'{prefix}*{suffix}' 패턴에 여러 파일이 걸림 — {[p.name for p in matches]}")
    raise FileNotFoundError(f"{pred_dir}/{prefix}*{suffix} 못 찾음")


def _read_preds(path: Path) -> dict:
    preds = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            preds[row["case_id"]] = (float(row["risk"]), float(row["OS_time"]), int(row["OS_event"]))
    return preds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True, choices=["tcga", "cptac", "brca"])
    parser.add_argument("--split", type=str, required=True, choices=["internal", "external"])
    parser.add_argument("--models", type=str, required=True, help="콤마로 구분한 model_prefix 목록(2개 이상)")
    parser.add_argument("--seeds", type=str, required=True, help="콤마로 구분한 seed 목록 — 모든 모델에 공통 적용")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()

    models = args.models.split(",")
    if len(models) < 2:
        raise ValueError("--models에 2개 이상의 태그를 콤마로 구분해 넘겨야 합니다.")
    seeds = [int(s) for s in args.seeds.split(",")]
    pred_dir = _ROOT / ".logs" / ("kfold_preds" if args.split == "internal" else "external_preds")

    # {case_id: [risk1, risk2, ...]} — model x seed(x fold) 조합 전부의 risk를 모은다.
    risk_lists: dict[str, list[float]] = {}
    ref_label: dict[str, tuple[float, int]] = {}
    n_sources = 0

    for model in models:
        for seed in seeds:
            # internal도 external과 마찬가지로 seed당 fold별 파일이 따로 있다(k-fold CV라
            # fold마다 다른 held-out 환자를 예측) — 둘 다 n_folds개 파일을 전부 읽어야 한다.
            for fold in range(args.n_folds):
                suffix = f"_seed{seed}_fold{fold}of{args.n_folds}.csv"
                path = _find_pred_path(pred_dir, args.dataset, model, suffix)
                preds = _read_preds(path)
                n_sources += 1
                for cid, (risk, t, e) in preds.items():
                    risk_lists.setdefault(cid, []).append(risk)
                    if cid in ref_label:
                        ref_t, ref_e = ref_label[cid]
                        if abs(t - ref_t) > 1e-6 or e != ref_e:
                            raise ValueError(f"case_id={cid}의 OS_time/OS_event가 소스마다 다름 — 라벨 불일치 의심")
                    else:
                        ref_label[cid] = (t, e)

    print(f"=== {len(models)}개 유전자셋 모델({models}) x {len(seeds)}시드 "
          f"({'internal, seed당 out-of-fold' if args.split == 'internal' else f'external, seed x {args.n_folds}fold'}) "
          f"교차 앙상블 — {args.split}, {args.dataset} ===")
    print(f"  소스(model x seed{'x fold' if args.split == 'external' else ''}) 개수: {n_sources}")

    # internal: 모든 소스(model x seed)가 이 환자를 held-out으로 커버해야 정당한 평균 —
    # 소스마다 case 집합이 다르면(예: 서로 다른 유전자셋이 학습 가능한 case pool을 다르게 필터링)
    # 교집합만 쓴다.
    common_cases = sorted(ref_label.keys())
    n_per_case = {cid: len(risk_lists[cid]) for cid in common_cases}
    max_n = max(n_per_case.values())
    incomplete = [cid for cid, n in n_per_case.items() if n != max_n]
    if incomplete:
        print(f"  [경고] {len(incomplete)}명은 전체 소스({max_n}개)를 다 못 채움 — 있는 것만으로 평균")

    ensembled_risks = np.array([np.mean(risk_lists[c]) for c in common_cases])
    times = np.array([ref_label[c][0] for c in common_cases])
    events = np.array([ref_label[c][1] for c in common_cases])

    m = compute_survival_metrics(ensembled_risks, times, events)
    print(f"\n=== 유전자셋 교차 앙상블 결과 (N={len(common_cases)}, events={int(events.sum())}) ===")
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
        print(f"  bootstrap 95% CI (환자 단위 resample, n={args.bootstrap}) = [{lo:.4f}, {hi:.4f}], std={boot_c.std():.4f}")


if __name__ == "__main__":
    main()
