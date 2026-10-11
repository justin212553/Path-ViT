"""
2026-10-10: 리뷰 대응([정찬권 #9], SciRep Q2) — fold-safe 배치 예측에 HR, IBS, calibration 추가.
재학습 없이 저장된 앙상블 risk만 쓴다(앙상블/코호트 정의는 analyze_foldsafe_batch.py의 load).

1) HR
   - median-split HR: 고위험 = risk > internal median. external도 internal median을 cutoff로 써서
     external 정보를 쓰지 않는다(analyze_foldsafe_batch의 logrank는 각 집합 자체 median이라 다름)
   - 연속 HR: internal 평균/표준편차로 표준화한 risk 1 SD당 HR
2) risk -> 생존확률 변환(재보정): internal OOF 앙상블에서 Cox(z) 적합(β + Breslow baseline).
   external은 이 변환을 그대로 적용. internal 성능은 같은 데이터로 1-모수 재보정을 한 값이라 약간 낙관적
3) IBS: IPCW Brier score를 (t0, tau]에서 적분, 검열 분포 G는 평가 집합 자체 KM(Uno's C와 같은 관례).
   기준선: internal KM 곡선을 모든 환자에게 똑같이 준 null 모델. scaled = 1 − IBS/IBS_null
4) calibration: 시점 t에서 예측 S(t) 5분위 그룹별 평균 예측 vs KM 관측 생존. external calibration slope =
   external 결과에 linear predictor(β·z)로 Cox 적합한 계수(이상적 1). 대표 모델 그림 PNG

출력: paper/results_hr_ibs_calib.md, _models.tsv, paper/figures/calibration_{cohort}.png
사용법: python scripts/analyze_hr_ibs_calib.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from analyze_foldsafe_batch import COHORTS, load  # noqa: E402
from analyze_uno_tdauc import censor_km_left  # noqa: E402

TAU = {"PAAD": 1095.0, "BRCA": 1825.0}
T0 = 30.0
CAL_TIMES = {"PAAD": [365.0, 730.0], "BRCA": [1095.0, 1825.0]}
N_GROUPS = 5
PLOT_MODELS = {"PAAD": ["M4", "M7", "M5"], "BRCA": ["M4", "M7", "M5"]}
SERIES = ["#2a78d6", "#eb6834"]  # dataviz 기본 팔레트 categorical 1·2 (light)
MARKERS = ["o", "s"]
AX_LO = {"PAAD": 0.0, "BRCA": 0.4}  # BRCA는 생존이 높아 점이 0.6–1에 몰림 — 축을 좁혀 분위 간 차이를 보이게


def cox_hr(df, col):
    cph = CoxPHFitter().fit(df[[col, "T", "E"]], "T", "E")
    s = cph.summary.loc[col]
    return float(s["exp(coef)"]), float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"]), float(s["p"])


def frame(r, t, e):
    return pd.DataFrame({"r": r, "T": np.asarray(t, float), "E": np.asarray(e, int)})


def surv_at(cph, z, times):
    """S(t|z) 행렬 (환자 x 시점)."""
    sf = cph.predict_survival_function(pd.DataFrame({"z": z}), times=times)
    return sf.to_numpy().T


def ibs(S, t, e, grid):
    """IPCW integrated Brier score (Graf). S: 환자 x grid."""
    g_t = censor_km_left(t, e, t)
    g_grid = censor_km_left(t, e, grid + 1e-9)  # G(s), s 시점 생존한 사람 가중
    bs = []
    for j, s in enumerate(grid):
        died = (t <= s) & (e == 1)
        alive = t > s
        w_d = np.where(died & (g_t > 0), 1.0 / np.maximum(g_t, 1e-12), 0.0)
        w_a = np.where(alive, 1.0 / max(g_grid[j], 1e-12), 0.0)
        bs.append(np.mean(w_d * S[:, j] ** 2 + w_a * (1 - S[:, j]) ** 2))
    bs = np.asarray(bs)
    return np.trapezoid(bs, grid) / (grid[-1] - grid[0])


def calib_groups(S_t, t, e, at):
    q = pd.qcut(S_t, N_GROUPS, labels=False, duplicates="drop")
    rows = []
    for g in sorted(np.unique(q)):
        m = q == g
        km = KaplanMeierFitter().fit(t[m], e[m])
        obs = float(km.predict(at))
        ci = km.confidence_interval_survival_function_
        idx = ci.index.searchsorted(at, side="right") - 1
        lo, hi = (float(ci.iloc[max(idx, 0), 0]), float(ci.iloc[max(idx, 0), 1]))
        rows.append((float(S_t[m].mean()), obs, lo, hi, int(m.sum())))
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _ROOT / "paper"
    (out / "figures").mkdir(exist_ok=True)
    rows, calib = [], {}
    for cohort, c in COHORTS.items():
        grid = np.linspace(T0, TAU[cohort], 60)
        for name, tag in c["models"].items():
            ids_i, r_i, t_i, e_i = load(cohort, "internal", tag)
            ids_e, r_e, t_e, e_e = load(cohort, "external", tag)
            t_i, e_i, t_e, e_e = (np.asarray(x, float if k % 2 == 0 else int)
                                  for k, x in enumerate((t_i, e_i, t_e, e_e)))
            mu, sd = r_i.mean(), r_i.std()
            cut = np.median(r_i)
            row = {"cohort": cohort, "model": name}
            fit = None
            for split, r, t, e in (("internal", r_i, t_i, e_i), ("external", r_e, t_e, e_e)):
                df = frame(r, t, e)
                df["high"] = (df.r > cut).astype(int)
                df["z"] = (df.r - mu) / sd
                row[f"{split}_hr_median"], row[f"{split}_hr_median_lo"], row[f"{split}_hr_median_hi"], _ = cox_hr(df, "high")
                hi_ = df.high == 1
                row[f"{split}_logrank_p"] = float(logrank_test(df["T"][hi_], df["T"][~hi_],
                                                               df["E"][hi_], df["E"][~hi_]).p_value)
                row[f"{split}_n_high"] = int(df.high.sum())
                # 자기 집합 median cutoff — 코호트 간 risk 척도 이동(PAAD RNA 모델은 CPTAC이 통째로 고위험 쪽으로
                # 밀림)에 영향받지 않는 상대 순위 기준 HR. 주 보고값
                df["high_own"] = (df.r > df.r.median()).astype(int)
                row[f"{split}_hr_own"], row[f"{split}_hr_own_lo"], row[f"{split}_hr_own_hi"], _ = cox_hr(df, "high_own")
                ho = df.high_own == 1
                row[f"{split}_logrank_own_p"] = float(logrank_test(df["T"][ho], df["T"][~ho],
                                                                   df["E"][ho], df["E"][~ho]).p_value)
                row[f"{split}_hr_sd"], row[f"{split}_hr_sd_lo"], row[f"{split}_hr_sd_hi"], row[f"{split}_hr_sd_p"] = cox_hr(df, "z")
                if split == "internal":
                    fit = CoxPHFitter().fit(df[["z", "T", "E"]], "T", "E")
                    km_null = KaplanMeierFitter().fit(t, e)
                S = surv_at(fit, df.z.to_numpy(), grid)
                S_null = np.tile(km_null.predict(grid).to_numpy(), (len(t), 1))
                row[f"{split}_ibs"] = ibs(S, t, e, grid)
                row[f"{split}_ibs_null"] = ibs(S_null, t, e, grid)
                row[f"{split}_ibs_scaled"] = 1 - row[f"{split}_ibs"] / row[f"{split}_ibs_null"]
                if split == "external":
                    lp = fit.params_["z"] * df.z
                    cal = CoxPHFitter().fit(pd.DataFrame({"lp": lp, "T": t, "E": e}), "T", "E")
                    s = cal.summary.loc["lp"]
                    row["external_cal_slope"] = float(s["coef"])
                    row["external_cal_slope_lo"] = float(s["coef lower 95%"])
                    row["external_cal_slope_hi"] = float(s["coef upper 95%"])
                for at in CAL_TIMES[cohort]:
                    S_t = surv_at(fit, df.z.to_numpy(), [at])[:, 0]
                    km_all = float(KaplanMeierFitter().fit(t, e).predict(at))
                    row[f"{split}_pred_{int(at / 365)}y"] = float(S_t.mean())
                    row[f"{split}_obs_{int(at / 365)}y"] = km_all
                    if name in PLOT_MODELS[cohort]:
                        calib[(cohort, name, split, at)] = calib_groups(S_t, t, e, at)
            rows.append(row)
            print(f"{cohort} {name:11s} ext HR(median) {row['external_hr_median']:.2f} IBS {row['external_ibs']:.4f} "
                  f"(null {row['external_ibs_null']:.4f}) slope {row['external_cal_slope']:.2f}", flush=True)

    m = pd.DataFrame(rows)
    m.to_csv(out / "results_hr_ibs_calib_models.tsv", sep="\t", index=False)

    # --- 그림: 코호트별, 행=split, 열=모델, 계열=시점 ---
    for cohort in COHORTS:
        models = PLOT_MODELS[cohort]
        fig, axes = plt.subplots(2, len(models), figsize=(3.2 * len(models), 6.4), sharex=True, sharey=True,
                                 facecolor="#fcfcfb")
        for i, split in enumerate(("internal", "external")):
            for j, name in enumerate(models):
                ax = axes[i, j]
                ax.set_facecolor("#fcfcfb")
                ax.plot([AX_LO[cohort], 1], [AX_LO[cohort], 1], color="#9a9890", lw=1, ls="--", zorder=1)
                for k, at in enumerate(CAL_TIMES[cohort]):
                    g = calib[(cohort, name, split, at)]
                    x = [p[0] for p in g]; y = [p[1] for p in g]
                    yerr = [[p[1] - p[2] for p in g], [p[3] - p[1] for p in g]]
                    ax.errorbar(x, y, yerr=yerr, color=SERIES[k], marker=MARKERS[k], ms=6, lw=2, capsize=2,
                                mec="#fcfcfb", mew=1.5, label=f"{int(at / 365)}-year", zorder=3)
                ax.set_xlim(AX_LO[cohort], 1); ax.set_ylim(AX_LO[cohort], 1)
                ax.grid(color="#e4e3dd", lw=0.6); ax.set_axisbelow(True)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.tick_params(colors="#52514e", labelsize=8)
                ax.set_title(f"{name} — {split}", fontsize=10, color="#0b0b0b")
                if i == 1:
                    ax.set_xlabel("Predicted survival", fontsize=9, color="#52514e")
                if j == 0:
                    ax.set_ylabel("Observed survival (KM)", fontsize=9, color="#52514e")
        axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.suptitle(f"{cohort} calibration — risk quintiles", fontsize=11, color="#0b0b0b")
        fig.tight_layout()
        fig.savefig(out / "figures" / f"calibration_{cohort.lower()}.png", dpi=200, facecolor="#fcfcfb")
        plt.close(fig)

    # --- 보고서 ---
    L = ["# HR / IBS / calibration (fold-safe 배치, 2026-10-10)", "",
         "앙상블은 `results_foldsafe_batch.md`와 동일(5 seeds x 5 folds, best+final 평균). "
         "median-split HR(주): 각 집합 자체 median으로 상/하 절반. 참고: internal median을 external에 그대로 쓴 HR과 그때 "
         "고위험군 인원 — 코호트 간 risk 척도가 이동하면 한쪽으로 쏠림. 연속 HR은 internal 기준 1 SD당. "
         f"risk→S(t) 변환은 internal OOF에서 Cox(z) 재보정 후 external에 그대로 적용. IBS 구간 {T0:.0f}일~tau "
         "(PAAD 3년, BRCA 5년), IPCW, 검열 분포는 평가 집합 KM. null = internal KM 곡선. "
         "scaled IBS = 1 − IBS/IBS_null(클수록 좋음, 0이면 null과 같음). internal IBS/calibration은 같은 데이터로 "
         "재보정해 약간 낙관적.", ""]
    for cohort in COHORTS:
        x = m[m.cohort == cohort]
        L += [f"## {cohort} — HR", "",
              "| model | internal HR median-split [95% CI] | external HR median-split [95% CI] | external logrank p | "
              "internal HR/SD | external HR/SD [95% CI] | 참고: internal cutoff 적용 시 external 고위험 n / HR |",
              "|---|---|---|---|---|---|---|"]
        for _, r in x.iterrows():
            L.append(f"| {r.model} | {r.internal_hr_own:.2f} [{r.internal_hr_own_lo:.2f}, {r.internal_hr_own_hi:.2f}] | "
                     f"{r.external_hr_own:.2f} [{r.external_hr_own_lo:.2f}, {r.external_hr_own_hi:.2f}] | "
                     f"{'<0.0001' if r.external_logrank_own_p < 1e-4 else f'{r.external_logrank_own_p:.4f}'} | "
                     f"{r.internal_hr_sd:.2f} | {r.external_hr_sd:.2f} [{r.external_hr_sd_lo:.2f}, {r.external_hr_sd_hi:.2f}] | "
                     f"{r.external_n_high} / {r.external_hr_median:.2f} |")
        L += ["", f"## {cohort} — IBS와 calibration", "",
              "| model | internal IBS (scaled) | external IBS | external IBS null | external scaled | external cal. slope [95% CI] | "
              + " | ".join(f"ext 예측/관측 {int(a / 365)}y" for a in CAL_TIMES[cohort]) + " |",
              "|---|---|---|---|---|---|" + "---|" * len(CAL_TIMES[cohort])]
        for _, r in x.iterrows():
            L.append(f"| {r.model} | {r.internal_ibs:.4f} ({r.internal_ibs_scaled:+.3f}) | {r.external_ibs:.4f} | "
                     f"{r.external_ibs_null:.4f} | {r.external_ibs_scaled:+.3f} | "
                     f"{r.external_cal_slope:.2f} [{r.external_cal_slope_lo:.2f}, {r.external_cal_slope_hi:.2f}] | "
                     + " | ".join(f"{r[f'external_pred_{int(a / 365)}y']:.3f} / {r[f'external_obs_{int(a / 365)}y']:.3f}"
                                  for a in CAL_TIMES[cohort]) + " |")
        L += ["", f"그림: `paper/figures/calibration_{cohort.lower()}.png` ({', '.join(PLOT_MODELS[cohort])}, "
                  f"{' / '.join(f'{int(a / 365)}년' for a in CAL_TIMES[cohort])}, 예측 생존 5분위)", ""]
    (out / "results_hr_ibs_calib.md").write_text("\n".join(L), encoding="utf-8")
    print("written paper/results_hr_ibs_calib.md")


if __name__ == "__main__":
    main()
