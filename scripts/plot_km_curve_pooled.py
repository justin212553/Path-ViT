"""
2seed x 5fold(+FINALEPOCH 체크포인트) pooled 앙상블 예측으로 Kaplan-Meier 곡선을 그린다.

scripts/plot_km_curve.py는 fold/seed 하나짜리 CSV 한 장만 그리는데, 논문이 실제로 보고하는
수치는 항상 scripts/pool_multiseed_kfold_preds.py(internal)와 scripts/
pool_multiseed_external_preds.py(external)로 만든 pooled 앙상블이다. 그림도 그 수치와 같은
risk score로 그려야 하므로, 이 두 스크립트의 pooling 함수(_load_seed_predictions /
_load_run_predictions)를 그대로 import해서 재사용한다 — 둘 다 pred_dir을 인자로 받게 이미
작성돼 있어서, 로직을 새로 베끼지 않고 paper/m1_m7_final_ablation_preds 같은 다른 디렉터리를
그대로 넘길 수 있다.

internal: seed별로 5-fold pooled out-of-fold를 만든 뒤, 같은 환자에 대한 여러 seed의 예측을
평균한다(각 seed의 예측은 항상 그 환자를 한 번도 학습에 쓰지 않은 모델에서 나옴 — held-out
원칙 유지).
external: 이 코호트는 어느 (seed, fold) checkpoint로 평가해도 학습에 전혀 등장하지 않으므로,
2seed x 5fold 체크포인트 전부의 예측을 환자 단위로 그대로 평균한다.

사용법:
    # internal (TCGA-PAAD, M4, pooled)
    python scripts/plot_km_curve_pooled.py --dataset tcga \\
        --model M4_uni2native_INT1500_SS_STG_R_MUT_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \\
        --split internal --pred-dir paper/m1_m7_final_ablation_preds/kfold_preds \\
        --include-final-epoch --title "M4 Internal (TCGA-PAAD, pooled)" \\
        --out paper/figures/km_m4_internal_pooled.png

    # external (CPTAC-PDAC, M4, pooled)
    python scripts/plot_km_curve_pooled.py --dataset cptac \\
        --model M4_uni2native_INT1500_SS_STG_R_MUT_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \\
        --split external --pred-dir paper/m1_m7_final_ablation_preds/external_preds \\
        --include-final-epoch --title "M4 External (CPTAC, pooled)" \\
        --out paper/figures/km_m4_external_pooled.png

model_prefix는 파일명에서 "{dataset}_" 다음, "_FOLD{i}OF{n}..." 앞부분 그대로(예:
paper/m1_m7_final_ablation_preds/kfold_preds/의 실제 파일명을 보고 맞출 것).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.pool_multiseed_kfold_preds import _load_seed_predictions as _load_internal_seed_preds
from scripts.pool_multiseed_external_preds import _load_run_predictions as _load_external_run_preds


def _pool_internal(pred_dir, dataset, model, seeds, n_folds, include_final_epoch):
    per_seed_preds = {
        seed: _load_internal_seed_preds(pred_dir, dataset, model, seed, n_folds, include_final_epoch=include_final_epoch)
        for seed in seeds
    }
    case_sets = [set(p.keys()) for p in per_seed_preds.values()]
    common = set.intersection(*case_sets)
    if any(cs != common for cs in case_sets):
        missing = set.union(*case_sets) - common
        print(f"  [경고] 일부 case가 모든 seed에 있지 않음({len(missing)}명) — 교집합({len(common)}명)만 사용")
    common = sorted(common)

    risks, times, events = [], [], []
    for cid in common:
        seed_risks, ref_t, ref_e = [], None, None
        for seed in seeds:
            r, t, e = per_seed_preds[seed][cid]
            seed_risks.append(r)
            if ref_t is None:
                ref_t, ref_e = t, e
            elif abs(t - ref_t) > 1e-6 or e != ref_e:
                raise ValueError(f"case_id={cid} 라벨이 seed마다 다름 — 데이터 불일치 의심")
        risks.append(float(np.mean(seed_risks)))
        times.append(ref_t)
        events.append(ref_e)
    return np.array(risks), np.array(times), np.array(events)


def _pool_external(pred_dir, dataset, model, seeds, n_folds, include_final_epoch):
    patient_risks: dict[str, list] = defaultdict(list)
    patient_label: dict[str, tuple] = {}
    for seed in seeds:
        for fold in range(n_folds):
            preds = _load_external_run_preds(
                pred_dir, dataset, model, seed, fold, n_folds, include_final_epoch=include_final_epoch,
            )
            for cid, (r, t, e) in preds.items():
                patient_risks[cid].append(r)
                if cid in patient_label and patient_label[cid] != (t, e):
                    raise ValueError(f"case_id={cid} 라벨이 실행마다 다름 — 데이터 불일치 의심")
                patient_label[cid] = (t, e)
    case_ids = sorted(patient_risks.keys())
    risks = np.array([float(np.mean(patient_risks[c])) for c in case_ids])
    times = np.array([patient_label[c][0] for c in case_ids])
    events = np.array([patient_label[c][1] for c in case_ids])
    return risks, times, events


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True, choices=["tcga", "cptac", "brca"])
    parser.add_argument("--model", type=str, required=True,
                         help="model_prefix, 예: M4_uni2native_INT1500_SS_STG_R_MUT_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1")
    parser.add_argument("--split", type=str, required=True, choices=["internal", "external"])
    parser.add_argument("--pred-dir", type=str, required=True,
                         help="예: paper/m1_m7_final_ablation_preds/kfold_preds 또는 .../external_preds")
    parser.add_argument("--seeds", type=str, default="84,126")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--include-final-epoch", action="store_true",
        help="이 연구 최종 관례 — best-checkpoint와 FINALEPOCH 예측을 환자 단위로 평균해서 쓴다. "
             "논문 표 수치와 맞추려면 켤 것.",
    )
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.split == "internal":
        risk, time, event = _pool_internal(
            pred_dir, args.dataset, args.model, seeds, args.n_folds, args.include_final_epoch
        )
    else:
        risk, time, event = _pool_external(
            pred_dir, args.dataset, args.model, seeds, args.n_folds, args.include_final_epoch
        )

    # utils/metrics.py::compute_survival_metrics와 동일 관례: median split
    high_risk = risk > np.median(risk)
    lr = logrank_test(
        time[high_risk], time[~high_risk],
        event_observed_A=event[high_risk], event_observed_B=event[~high_risk],
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(time[~high_risk], event[~high_risk], label=f"Low risk (n={int((~high_risk).sum())})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#2166ac")
    kmf.fit(time[high_risk], event[high_risk], label=f"High risk (n={int(high_risk.sum())})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#b2182b")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Overall survival probability")
    ax.set_ylim(0, 1.05)
    default_title = f"{args.model} ({args.split}, pooled {len(seeds)}seed x {args.n_folds}fold)"
    ax.set_title(args.title or default_title)
    ax.text(0.98, 0.95, f"log-rank p = {lr.p_value:.4f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10)
    ax.legend(loc="lower left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"N={len(risk)}, events={int(event.sum())}, log-rank p={lr.p_value:.4f}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
