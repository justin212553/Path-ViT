"""
PORPOISE 공식 코드(porpoise/main.py) 5-fold CV 결과를 이 프로젝트 관례(pooled out-of-fold —
scripts/pool_multiseed_kfold_preds.py와 동일한 정신)로 재집계한다.

PORPOISE 자기 자신의 summary_latest.csv는 fold별 c-index의 단순 산술평균만 준다(작은 fold
표본 하나하나의 분산이 그대로 드러남 — 우리 쪽 실험에서도 fold별 HR이 수천만 단위로 튀는 걸
실제로 겪었다). 이 프로젝트는 처음부터 "5개 fold의 held-out 예측을 전부 합쳐서(각 환자는
정확히 한 fold에서만 held-out — 5-fold CV라 겹침 없음) 그 위에서 c-index/HR/log-rank를 한 번에
다시 계산"하는 쪽을 표준으로 써왔다 — 표본이 커져서 median-split HR 계산도 훨씬 안정적이다.

porpoise/utils/core_utils.py::summary_survival이 각 fold의 val(held-out) 환자별 예측을
patient_results(dict: case_id -> {risk, disc_label, survival, censorship})로 반환하고,
main.py가 이를 <results_dir>/.../split_latest_val_{fold}_results.pkl로 저장한다(순수 pickle,
porpoise 패키지 import 불필요) — 이 5개 pkl을 찾아 합친다.

**주의(극성)**: PORPOISE는 censorship=1이 "censored"(이 프로젝트 event=1="사망"과 정반대) —
event = 1 - censorship로 변환해서 utils.metrics.compute_survival_metrics(이 프로젝트 표준
지표 함수, cox_ph_loss와 동일한 event 극성 기대)에 넘긴다.

사용법:
    python scripts/pool_porpoise_official_kfold.py --results-dir porpoise/results_true_resnet50_mmf
    python scripts/pool_porpoise_official_kfold.py --results-dir porpoise/results_true_resnet50_mmf --bootstrap 2000
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="PORPOISE main.py --results_dir로 준 최상위 경로(예: porpoise/results_true_resnet50_mmf) "
             "— 정확한 하위 param_code/exp_code 폴더명을 몰라도 재귀적으로 "
             "split_latest_val_*_results.pkl을 전부 찾는다. 여러 실행(다른 --seed 등)의 결과가 "
             "섞여 있으면 전부 합쳐지므로, 한 실행분만 보려면 그 실행의 정확한 하위 폴더를 넘길 것.",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=0,
        help="주어지면(예: 2000) 환자 단위 resample bootstrap 95%% CI를 추가로 계산한다. 0(기본)이면 생략.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    pkl_paths = sorted(results_dir.glob("**/split_latest_val_*_results.pkl"))
    if not pkl_paths:
        raise FileNotFoundError(f"{results_dir} 아래에서 split_latest_val_*_results.pkl을 못 찾음")
    print(f"발견된 fold 결과 pkl {len(pkl_paths)}개:")
    for p in pkl_paths:
        print(f"  {p}")

    risks, times, events, case_ids = [], [], [], []
    for p in pkl_paths:
        with open(p, "rb") as f:
            patient_results = pickle.load(f)
        for cid, d in patient_results.items():
            risks.append(float(d["risk"]))
            times.append(float(d["survival"]))
            events.append(1.0 - float(d["censorship"]))  # 극성 반전(위 docstring 참조)
            case_ids.append(cid)

    n_unique = len(set(case_ids))
    if n_unique != len(case_ids):
        print(f"  [경고] case_id 중복 {len(case_ids) - n_unique}개 — 여러 실행이 섞였거나 "
              f"fold가 겹쳤을 가능성(정상 5-fold CV라면 안 나와야 함)")

    risks  = np.array(risks)
    times  = np.array(times)
    events = np.array(events)

    print(f"\n=== pooled out-of-fold({len(pkl_paths)}-fold 전체, N={len(case_ids)}, "
          f"events={int(events.sum())}) ===")
    m = compute_survival_metrics(risks, times, events)
    print(f"  c_index={m['c_index']:.4f} | HR={m['hr']:.3f} [{m['hr_ci_lower']:.3f}, {m['hr_ci_upper']:.3f}] | "
          f"log_rank_p={m['log_rank_p']:.4f}")

    if args.bootstrap > 0:
        rng = np.random.RandomState(0)
        n = len(case_ids)
        boot_c = []
        for _ in range(args.bootstrap):
            idx = rng.randint(0, n, n)
            bm = compute_survival_metrics(risks[idx], times[idx], events[idx])
            if not np.isnan(bm["c_index"]):
                boot_c.append(bm["c_index"])
        boot_c = np.array(boot_c)
        lo, hi = np.percentile(boot_c, [2.5, 97.5])
        print(f"  bootstrap 95% CI (환자 단위 resample, n={args.bootstrap}) = "
              f"[{lo:.4f}, {hi:.4f}], std={boot_c.std():.4f}")


if __name__ == "__main__":
    main()
