"""
PORPOISE 공식 코드(porpoise/main.py) 5-fold CV 결과를, 여러 --seed 반복까지 포함해서 이
프로젝트 관례(scripts/pool_multiseed_kfold_preds.py와 동일한 정신 — seed별 pooled out-of-fold
지표 + 환자 단위 seed 간 예측 평균 앙상블)로 재집계한다.

PORPOISE 자기 자신의 summary_latest.csv는 fold별 c-index의 단순 산술평균만 주고, 논문도 seed
하나짜리 5-fold CV 한 번만 보고한다(Methods 원문 확인 — "We repeated the experiments ... in a
five-fold cross-validation ... five times"뿐, 여러 시드 반복/평균 언급 없음). 이 프로젝트는
모든 자체 모델을 2seed(84,126) x 5fold pooled+앙상블로 보고해왔으므로, PORPOISE 쪽도 동일한
잣대(apple-to-apple)로 비교하려면 최소 2개 이상의 --seed로 반복해서 이 스크립트로 합쳐야 한다.

fold 분할 자체(어느 case가 어느 fold인지)는 PORPOISE 공식 splits/5foldcv/tcga_paad/splits_{i}.csv
(seed 무관 고정 파일)를 그대로 쓰므로, seed마다 fold 배정이 달라지는 우리 자체 모델과 달리
"seed 간 예측 평균"이 held-out 원칙을 위반하지 않는다는 보장은 없다(같은 fold 배정이면 한
환자가 어느 seed에서도 항상 같은 fold에서 held-out되므로 오히려 더 깨끗함) — 그래도 평균 방식
자체는 pool_multiseed_kfold_preds.py와 동일하게 유지한다(값 자체가 다른 모델 초기화에서 나온
독립적 추정치라는 점은 같음).

porpoise/utils/core_utils.py::summary_survival이 각 fold의 val(held-out) 환자별 예측을
patient_results(dict: case_id -> {risk, disc_label, survival, censorship})로 반환하고,
main.py가 <results_dir>/<which_splits>/<param_code>/<exp_code>_s{seed}/split_latest_val_
{fold}_results.pkl로 저장한다(순수 pickle, porpoise 패키지 import 불필요) — 시드별 하위 폴더가
"_s{seed}"로 끝나는 것을 이용해 시드별로 모은다.

**주의(극성)**: PORPOISE는 censorship=1이 "censored"(이 프로젝트 event=1="사망"과 정반대) —
event = 1 - censorship로 변환해서 utils.metrics.compute_survival_metrics에 넘긴다.

사용법:
    python scripts/pool_porpoise_official_kfold.py --results-dir porpoise/results_true_resnet50_mmf --seeds 1
    python scripts/pool_porpoise_official_kfold.py --results-dir porpoise/results_true_resnet50_mmf --seeds 1,84 --bootstrap 2000
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.metrics import compute_survival_metrics


def _load_seed_predictions(results_dir: Path, seed: int) -> dict:
    """한 시드의 5-fold 전체 held-out 예측을 모아 {case_id: (risk, time, event)}로 반환한다."""
    pkl_paths = sorted(results_dir.glob(f"**/*_s{seed}/split_latest_val_*_results.pkl"))
    if not pkl_paths:
        raise FileNotFoundError(
            f"{results_dir} 아래에서 seed={seed} 결과(**/*_s{seed}/split_latest_val_*_results.pkl)를 못 찾음"
        )
    preds = {}
    for p in pkl_paths:
        with open(p, "rb") as f:
            patient_results = pickle.load(f)
        for cid, d in patient_results.items():
            if cid in preds:
                raise ValueError(f"seed={seed} 내에서 case_id 중복: {cid} — fold가 겹쳤을 가능성")
            preds[cid] = (float(d["risk"]), float(d["survival"]), 1.0 - float(d["censorship"]))
    return preds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="PORPOISE main.py --results_dir로 준 최상위 경로(예: porpoise/results_true_resnet50_mmf) "
             "— 하위 param_code/exp_code_s{seed} 폴더명을 몰라도 재귀 탐색으로 찾는다.",
    )
    parser.add_argument(
        "--seeds", type=str, default="1",
        help="콤마구분 seed 목록(PORPOISE main.py --seed에 준 값들, 기본 '1'=main.py 기본값 하나만).",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=0,
        help="주어지면(예: 2000) 최종 앙상블 지표에 환자 단위 resample bootstrap 95%% CI를 추가로 계산한다.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    seeds = [int(s) for s in args.seeds.split(",")]

    per_seed_preds = {}
    per_seed_c = []
    print(f"=== seed별 pooled out-of-fold — {results_dir} (seeds={seeds}) ===")
    for seed in seeds:
        preds = _load_seed_predictions(results_dir, seed)
        per_seed_preds[seed] = preds
        risks  = np.array([v[0] for v in preds.values()])
        times  = np.array([v[1] for v in preds.values()])
        events = np.array([v[2] for v in preds.values()])
        m = compute_survival_metrics(risks, times, events)
        per_seed_c.append(m["c_index"])
        print(f"  seed={seed}: N={len(preds)}, events={int(events.sum())} | "
              f"c_index={m['c_index']:.4f} | HR={m['hr']:.3f} [{m['hr_ci_lower']:.3f}, {m['hr_ci_upper']:.3f}] | "
              f"log_rank_p={m['log_rank_p']:.4f}")

    if len(seeds) > 1:
        per_seed_c = np.array(per_seed_c)
        print(f"  -> seed 간 pooled c-index: mean={per_seed_c.mean():.4f}, std={per_seed_c.std():.4f} "
              f"(이게 반복측정으로 본 '이 정도 흔들린다'는 불확실성 폭)")

    # seed 간 case_id 집합이 동일한지 확인
    case_sets = [set(p.keys()) for p in per_seed_preds.values()]
    common_cases = set.intersection(*case_sets)
    if any(cs != common_cases for cs in case_sets):
        missing = set.union(*case_sets) - common_cases
        print(f"  [경고] 일부 case가 모든 seed에 있지 않음({len(missing)}명) — "
              f"교집합({len(common_cases)}명)만 앙상블에 사용")
    common_cases = sorted(common_cases)

    # 환자 단위 risk score를 seed 간 평균
    ensembled_risks, times, events = [], [], []
    for cid in common_cases:
        seed_risks = []
        ref_time, ref_event = None, None
        for seed in seeds:
            r, t, e = per_seed_preds[seed][cid]
            seed_risks.append(r)
            if ref_time is None:
                ref_time, ref_event = t, e
            elif abs(t - ref_time) > 1e-6 or e != ref_event:
                raise ValueError(f"case_id={cid}의 survival/censorship이 seed마다 다름 — 라벨 불일치 의심")
        ensembled_risks.append(float(np.mean(seed_risks)))
        times.append(ref_time)
        events.append(ref_event)

    ensembled_risks = np.array(ensembled_risks)
    times = np.array(times)
    events = np.array(events)
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
