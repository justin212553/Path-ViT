"""
2026-10-22: 원고 Figure 3·4 KM 그림. 원고 표와 같은 위험 점수(analyze_foldsafe_batch.load, 5 seeds x 5 folds,
best+final 평균)와 같은 분할 규칙(평가 집합 자체의 중앙값, risk > median이 고위험 — utils.metrics와 동일)을 쓴다.
옛 그림(img/km_pdac.png, km_brca.png)은 옛 번호 M4·옛 BRCA holdout 설계(827/230)·"p = 0.0000" 표기였다.

  km_pdac.png: M7, A TCGA-PAAD internal / B CPTAC-PDA external
  km_brca.png: M7, A TCGA-BRCA internal / B held-out institutions
원고 번호 M7 = 분석 산출물의 옛 번호 M4. p는 P < 0.0001 규칙으로 표기.

사용법: python -m scripts.make_km_figures   -> paper/figures/km_pdac.png, km_brca.png (+ .pdf)
QC 재실행 후에는 analyze_foldsafe_batch의 태그를 _QC/_INSTCVQC로 바꾼 뒤 다시 실행.
"""
import sys
from pathlib import Path

import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from analyze_foldsafe_batch import COHORTS, load  # noqa: E402

FIGS = [
    ("km_pdac", "PAAD", [("internal", "A", "TCGA-PAAD, internal"), ("external", "B", "CPTAC-PDA, external")]),
    ("km_brca", "BRCA", [("internal", "A", "TCGA-BRCA, internal"), ("external", "B", "TCGA-BRCA, held-out institutions")]),
]
ANALYSIS_ID = "M4"   # 원고 M7
LOW, HIGH = "#2a78d6", "#d03b3b"
INK, MUTED, GRID, SURF = "#1f1f1d", "#52514e", "#e4e3dd", "#ffffff"


def fmt_p(p):
    return "log-rank P < 0.0001" if p < 1e-4 else f"log-rank P = {p:.4f}"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    for name, cohort, panels in FIGS:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), facecolor=SURF)
        for ax, (split, letter, title) in zip(axes, panels):
            _, r, t, e = load(cohort, split, COHORTS[cohort]["models"][ANALYSIS_ID])
            r, t, e = np.asarray(r, float), np.asarray(t, float), np.asarray(e, int)
            high = r > np.median(r)
            p = logrank_test(t[~high], t[high], e[~high], e[high]).p_value
            for mask, lab, col in ((~high, "Low risk", LOW), (high, "High risk", HIGH)):
                km = KaplanMeierFitter().fit(t[mask], e[mask], label=f"{lab}, n = {int(mask.sum())}")
                km.plot_survival_function(ax=ax, color=col, lw=1.6, ci_show=True, ci_alpha=0.15)
            ax.set_facecolor(SURF)
            ax.set_ylim(0, 1.02)
            ax.set_xlim(left=0)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            ax.set_xlabel("Time, days", color=MUTED)
            ax.set_ylabel("Overall survival probability", color=MUTED)
            ax.set_title(title, fontsize=8.5, color=INK, loc="left")
            ax.text(-0.16, 1.02, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom")
            ax.text(0.97, 0.95, fmt_p(p), transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK)
            ax.legend(loc="lower left", frameon=False, fontsize=7.5)
            print(f"{name} {split}: N={len(r)} events={int(e.sum())} low={int((~high).sum())} high={int(high.sum())} p={p:.6f}")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = _ROOT / "paper" / "figures" / name
        fig.savefig(out.with_suffix(".png"), dpi=300, facecolor=SURF)
        fig.savefig(out.with_suffix(".pdf"), facecolor=SURF)
        plt.close(fig)
        print("saved", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
