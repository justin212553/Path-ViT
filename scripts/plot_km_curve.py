"""
Kaplan-Meier 생존곡선 플롯 — utils/metrics.py의 median-split 관례(high_risk = risk >
median(risk))와 동일 기준으로 high/low risk 두 그룹을 나눠 그린다. 논문 그림용 정적 PNG.

사용법:
    python scripts/plot_km_curve.py --csv .logs/kfold_preds/tcga_..._fold0of5.csv --title "M4 Internal (fold 0)" --out paper/figures/km_m4_internal_fold0.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="case_id,risk,OS_time,OS_event 컬럼을 가진 예측 CSV")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    risks, times, events = [], [], []
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            risks.append(float(row["risk"]))
            times.append(float(row["OS_time"]))
            events.append(int(row["OS_event"]))
    risk = np.array(risks)
    time = np.array(times)
    event = np.array(events)

    # utils/metrics.py::compute_survival_metrics와 동일 관례: median split
    high_risk = risk > np.median(risk)

    lr = logrank_test(
        time[high_risk], time[~high_risk],
        event_observed_A=event[high_risk], event_observed_B=event[~high_risk],
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(time[~high_risk], event[~high_risk], label=f"Low risk (n={(~high_risk).sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#2166ac")
    kmf.fit(time[high_risk], event[high_risk], label=f"High risk (n={high_risk.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#b2182b")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Overall survival probability")
    ax.set_ylim(0, 1.05)
    ax.set_title(args.title or Path(args.csv).stem)
    ax.text(0.98, 0.95, f"log-rank p = {lr.p_value:.4f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10)
    ax.legend(loc="lower left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"N={len(risk)}, events={event.sum()}, log-rank p={lr.p_value:.4f}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
