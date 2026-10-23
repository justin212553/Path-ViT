"""
2026-10-20: 본문용 calibration 그림(2패널). 기존 analyze_hr_ibs_calib.py 그림(코호트당 6칸, 두 시점 겹침)은
Supplementary로 돌리고, 본문은 메시지 두 개만 보인다.
  A. CPTAC-PDA 2년 — molecular data가 있는 M7은 순위는 맞지만 위험을 통째로 높게 잡음(점이 대각선 위)
  B. held-out TCGA-BRCA 기관 5년 — 두 모델 모두 대각선을 따라감
비교 모델은 M7(WSI, molecular, clinicopathologic)과 M3(clinicopathologic). 분석 산출물의 옛 번호로는 M4와 M5.
각 모델의 코호트 전체 평균(예측 vs 관측 KM)을 큰 마름모로 따로 찍어 calibration-in-the-large를 바로 보이게 한다.
변환은 analyze_hr_ibs_calib.py와 같다(internal OOF에서 Cox(z) 재보정 + Breslow, external에 그대로 적용).

사용법: python -m scripts.make_calibration_figure  -> paper/figures/calibration_main.png (+ .pdf)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from analyze_foldsafe_batch import COHORTS, load  # noqa: E402
from analyze_hr_ibs_calib import calib_groups, surv_at  # noqa: E402

PANELS = [("PAAD", 730.0, "A", "CPTAC-PDA, 2-year survival", 0.0, 0.8),
          ("BRCA", 1825.0, "B", "Held-out TCGA-BRCA institutions, 5-year survival", 0.4, 1.0)]
MODELS = [("M4", "M7", "M7: WSI, molecular, clinicopathologic", "#2a78d6", "o", 13),   # (분석 ID, 원고 번호, 범례, 색, 표식, 평균점 크기)
          ("M5", "M3", "M3: clinicopathologic", "#eb6834", "s", 7)]
# 예측 범위가 이보다 좁으면 slope가 불안정하다(PAAD M3: 0.37–0.41에서 8.0) — 그림에는 적지 않는다
MIN_RANGE_FOR_SLOPE = 0.1
INK, MUTED, GRID, SURF = "#1f1f1d", "#52514e", "#e4e3dd", "#ffffff"


def external_calibration(cohort, tag, at):
    _, r_i, t_i, e_i = load(cohort, "internal", tag)
    _, r_e, t_e, e_e = load(cohort, "external", tag)
    mu, sd = r_i.mean(), r_i.std()
    zi, ze = (r_i - mu) / sd, (r_e - mu) / sd
    t_i, e_i, t_e, e_e = np.asarray(t_i, float), np.asarray(e_i, int), np.asarray(t_e, float), np.asarray(e_e, int)
    fit = CoxPHFitter().fit(pd.DataFrame({"z": zi, "T": t_i, "E": e_i}), "T", "E")
    S_t = surv_at(fit, ze, [at])[:, 0]
    groups = calib_groups(S_t, t_e, e_e, at)
    lp = fit.params_["z"] * ze
    slope = float(CoxPHFitter().fit(pd.DataFrame({"lp": lp, "T": t_e, "E": e_e}), "T", "E").params_["lp"])
    obs = float(KaplanMeierFitter().fit(t_e, e_e).predict(at))
    return groups, float(S_t.mean()), obs, slope


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5), facecolor=SURF)
    for ax, (cohort, at, letter, title, lo, hi) in zip(axes, PANELS):
        ax.set_facecolor(SURF)
        ax.plot([lo, hi], [lo, hi], color="#9a9890", lw=1, ls="--", zorder=1)
        notes = []
        for k, (tag, num, label, col, mk, dms) in enumerate(MODELS):
            g, pred, obs, slope = external_calibration(cohort, COHORTS[cohort]["models"][tag], at)
            x = [p[0] for p in g]; y = [p[1] for p in g]
            yerr = [[p[1] - p[2] for p in g], [p[3] - p[1] for p in g]]
            ax.errorbar(x, y, yerr=yerr, color=col, marker=mk, ms=6, lw=1.6, elinewidth=1, capsize=2,
                        mec=SURF, mew=1.2, zorder=3)
            # 두 모델의 평균점이 겹쳐도 둘 다 보이도록 크기가 다른 속 빈 마름모를 겹쳐 그린다
            ax.plot([pred], [obs], marker="D", ms=dms, mfc="none", mec=col, mew=2.2, zorder=5 + k)
            sl = f"slope {slope:.2f}, " if max(x) - min(x) >= MIN_RANGE_FOR_SLOPE else "narrow prediction range, "
            notes.append(f"{num}: {sl}mean predicted {pred:.2f} vs observed {obs:.2f}")
            print(f"{cohort} {num} (analysis {tag}): slope {slope:.3f} pred {pred:.3f} obs {obs:.3f} "
                  f"quintiles {[round(v, 3) for v in x]}")
        # 수치는 축 아래에 둔다 — 축 안 상자는 PAAD의 최저 분위 점과 y축 눈금을 가렸다
        ax.text(0.0, -0.15, "\n".join(notes), transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK,
                linespacing=1.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        ax.set_xlabel("Predicted survival", color=MUTED)
        ax.set_ylabel("Observed survival, Kaplan-Meier", color=MUTED)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.text(-0.17, 1.02, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom")
    handles = [Line2D([], [], color=c, marker=m, ms=6, lw=1.6, mec=SURF, label=l) for _, _, l, c, m, _ in MODELS]
    handles += [Line2D([], [], ls="none", marker="o", ms=5, color=MUTED, label="Risk quintile, 95% CI"),
                Line2D([], [], ls="none", marker="D", ms=8, mfc=SURF, mec=MUTED, mew=1.8, label="Cohort mean"),
                Line2D([], [], color="#9a9890", lw=1, ls="--", label="Perfect calibration")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 0.97))
    out = _ROOT / "paper" / "figures" / "calibration_main"
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor=SURF)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURF)
    print("saved", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
