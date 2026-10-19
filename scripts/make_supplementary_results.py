"""
2026-10-18: Supplementary Information의 결과 표를 분석 산출물에서 바로 LaTeX로 만든다.
QC 재실행 등으로 수치가 바뀌면 분석 스크립트를 다시 돌린 뒤 이 스크립트만 다시 실행하면 된다.

입력(paper/):
  results_uno_tdauc_models.tsv      Uno's C, time-dependent AUC
  results_hr_ibs_calib_models.tsv   HR, IBS, calibration
  results_multivariable_cox_models.tsv / results_multivariable_cox.md   다변량 Cox(요약 + M4·M3 전체 공변량)
  results_ksens.md                  K 민감도
  results_baselines.md              아키텍처 수준 PORPOISE 비교(입력 통일)
  results_power_coattn.txt          scripts/power_coattn_external.py 출력(검정력)
출력: paper/sections/supplementary_results.tex

논문 표기 규칙(콜론·세미콜론 없음, \\label/\\ref 없음, 짧은 문장)을 따른다.
사용법: python scripts/make_supplementary_results.py
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "paper"
OUT = P / "sections" / "supplementary_results.tex"

COHORT_NAME = {"PAAD": ("TCGA-PAAD", "CPTAC-PDAC", "External"), "BRCA": ("TCGA-BRCA", "held-out TCGA-BRCA institutions", "Held-out")}
MODEL_ORDER = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M3-SA", "M4-SA", "M3-noCNV", "M4-noCNV", "M6-noCNV",
               "M7-noCNV", "M4-noMUT", "M4-FIXC", "M4-SA-FIXC"]


def f4(x):
    return "---" if pd.isna(x) else f"{x:.4f}"


def f2(x):
    return "---" if pd.isna(x) else f"{x:.2f}"


def fp(x):
    return "---" if pd.isna(x) else ("$<$0.0001" if x < 0.0001 else f"{x:.4f}")


def ci(v, lo, hi, fmt=f2):
    return f"{fmt(v)} [{fmt(lo)}, {fmt(hi)}]"


def tex_escape(s):
    return (s.replace("≥", "$\\geq$").replace("−", "$-$").replace("→", "$\\to$").replace("->", "$\\to$")
            .replace("%", "\\%").replace("**", ""))


def table(header, rows, caption, align=None):
    align = align or ("l" + "c" * (len(header) - 1))
    wide = len(header) >= 6  # 좁은 표까지 \resizebox로 늘리면 글자가 커지므로 넓은 표만 줄인다
    open_tab = "\\begin{tabular}{@{}" + align + "@{}}"
    out = ["\\begin{table}[htbp]", "  \\centering", "  " + ("\\resizebox{\\textwidth}{!}{" + open_tab if wide else open_tab),
           "    \\toprule", "    " + " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "    \\midrule"]
    for r in rows:
        out.append("    \\addlinespace" if r is None else "    " + " & ".join(r) + " \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}}" if wide else "  \\end{tabular}", f"  \\caption{{{caption}}}", "\\end{table}", ""]
    return "\n".join(out)


def ordered(df):
    df = df.copy()
    df["_o"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
    return df.sort_values("_o")


def md_table(path, heading):
    """heading으로 시작하는 줄 다음의 첫 마크다운 표를 행 리스트로."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(heading))
    rows = []
    for l in lines[i + 1:]:
        if l.startswith("|"):
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "):
                rows.append(cells)
        elif rows:
            break
    return rows


def sec_uno(df):
    out = ["\\subsection*{Uno's C-index and Time-Dependent AUC}",
           "Uno's C-index is truncated at 3 years for the pancreatic cohorts and 5 years for TCGA-BRCA. The time-dependent AUC is the cumulative and dynamic AUC at the listed time points. Both use inverse-probability-of-censoring weights estimated on the evaluation set itself. Variants after M7 are the single-factor and diagnostic variants described in the main text.", ""]
    for coh, times in (("PAAD", [1, 2, 3]), ("BRCA", [1, 3, 5])):
        d = ordered(df[df.cohort == coh])
        ext = COHORT_NAME[coh][2]
        header = ["Model", "Internal Uno's C"] + [f"Internal AUC {t}y" for t in times] + [f"{ext} Uno's C"] + [f"{ext} AUC {t}y" for t in times]
        rows = [[r.model, f4(r.internal_uno)] + [f4(getattr(r, f"internal_auc_{t}y")) for t in times]
                + [f4(r.external_uno)] + [f4(getattr(r, f"external_auc_{t}y")) for t in times] for r in d.itertuples()]
        out.append(table(header, rows, f"Uno's C-index and time-dependent AUC for {COHORT_NAME[coh][0]}, internal and on {COHORT_NAME[coh][1]}."))
    return out


def sec_hr(df):
    out = ["\\subsection*{Hazard Ratio}",
           "The median-split hazard ratio divides each evaluation set at its own median risk score. The log-rank $p$-value compares the same two groups. The continuous hazard ratio is per standard deviation of the risk score, standardized with the internal mean and standard deviation.", ""]
    for coh in ("PAAD", "BRCA"):
        d = ordered(df[df.cohort == coh])
        ext = COHORT_NAME[coh][2]
        header = ["Model", "Internal HR [95\\% CI]", "Internal $p$", f"{ext} HR [95\\% CI]", f"{ext} $p$", "Internal HR per SD", f"{ext} HR per SD [95\\% CI]"]
        rows = [[r.model, ci(r.internal_hr_own, r.internal_hr_own_lo, r.internal_hr_own_hi), fp(r.internal_logrank_own_p),
                 ci(r.external_hr_own, r.external_hr_own_lo, r.external_hr_own_hi), fp(r.external_logrank_own_p),
                 f2(r.internal_hr_sd), ci(r.external_hr_sd, r.external_hr_sd_lo, r.external_hr_sd_hi)] for r in d.itertuples()]
        out.append(table(header, rows, f"Median-split and continuous hazard ratios for {COHORT_NAME[coh][0]}, internal and on {COHORT_NAME[coh][1]}."))
    return out


def sec_ibs(df):
    out = ["\\subsection*{Integrated Brier Score and Calibration}",
           "Risk scores are converted to survival probabilities by a Cox model fit to the pooled internal predictions, and the same conversion is applied to the external evaluation. The integrated Brier score is integrated from 30 days to 3 years for the pancreatic cohorts and to 5 years for TCGA-BRCA. The scaled score is $1 - \\mathrm{IBS}/\\mathrm{IBS}_{\\mathrm{null}}$, where the null model assigns every patient the internal Kaplan-Meier curve. The calibration slope is the Cox coefficient of the converted linear predictor on the external outcomes, with 1 as ideal. Predicted survival is the mean over patients at each time point, and observed survival is the Kaplan-Meier estimate.", ""]
    for coh, (t1, t2) in (("PAAD", ("1y", "2y")), ("BRCA", ("3y", "5y"))):
        d = ordered(df[df.cohort == coh])
        ext = COHORT_NAME[coh][2]
        header = ["Model", "Internal IBS", "Internal scaled", f"{ext} IBS", f"{ext} scaled", "Calibration slope [95\\% CI]",
                  f"Predicted / observed {t1[0]}y", f"Predicted / observed {t2[0]}y"]
        rows = [[r.model, f4(r.internal_ibs), f"{r.internal_ibs_scaled:+.3f}", f4(r.external_ibs), f"{r.external_ibs_scaled:+.3f}",
                 ci(r.external_cal_slope, r.external_cal_slope_lo, r.external_cal_slope_hi),
                 f"{getattr(r, 'external_pred_' + t1):.3f} / {getattr(r, 'external_obs_' + t1):.3f}",
                 f"{getattr(r, 'external_pred_' + t2):.3f} / {getattr(r, 'external_obs_' + t2):.3f}"] for r in d.itertuples()]
        out.append(table(header, rows, f"Integrated Brier score and calibration for {COHORT_NAME[coh][0]}, internal and on {COHORT_NAME[coh][1]}. Calibration columns refer to the external evaluation."))
    return out


def sec_cox(df):
    out = ["\\subsection*{Multivariable Cox Analysis}",
           "Each model's risk score, per standard deviation, is adjusted for established clinicopathologic factors. For the pancreatic cohorts these are age per 10 years, sex, node-positive disease, and T category 3 or higher. For TCGA-BRCA they are age per 10 years, the ordinal T and N categories, and distant metastasis. Patients with a missing or indeterminate staging field are excluded from this analysis only. The likelihood-ratio $p$-value compares the model with and without the risk score.", ""]
    rows = []
    for coh in ("PAAD", "BRCA"):
        for r in df[df.cohort == coh].itertuples():
            split = "Internal" if r.split == "internal" else COHORT_NAME[coh][2]
            rows.append([COHORT_NAME[coh][0], r.model, split, f"{int(r.n)}", f"{int(r.events)}", ci(r.hr, r.lo, r.hi), fp(r.p), fp(r.lr_p)])
        rows.append(None)
    out.append(table(["Cohort", "Model", "Evaluation", "$N$", "Events", "Adjusted HR per SD [95\\% CI]", "$p$", "Likelihood-ratio $p$"],
                     rows[:-1], "Adjusted hazard ratio of each model's risk score in the multivariable Cox analysis.", align="lllcccc" + "c"))
    for coh in ("PAAD", "BRCA"):
        for m in ("M4", "M3"):
            rows = md_table(P / "results_multivariable_cox.md", f"### {coh} {m}")
            body = [[tex_escape(c[0]), tex_escape(c[1]), tex_escape(c[2])] for c in rows[1:]]
            body = [[b[0]] + [re.sub(r"\] (\S+)$", r"], $p$ = \1", x) for x in b[1:]] for b in body]
            out.append(table(["Covariate", "Internal HR [95\\% CI]", f"{COHORT_NAME[coh][2]} HR [95\\% CI]"], body,
                             f"All covariates of the multivariable Cox model for {m} in {COHORT_NAME[coh][0]}.", align="lcc"))
    return out


def sec_ksens():
    out = ["\\subsection*{Number of Clusters}",
           "M4 was retrained with $K=5$ and $K=20$ cluster centroids under the same seeds, folds, and fold-specific centroid fitting as the main $K=11$ model.", ""]
    for coh in ("PAAD", "BRCA"):
        rows = md_table(P / "results_ksens.md", f"## {coh} — 모델별")
        ext = COHORT_NAME[coh][2]
        out.append(table(["$K$", "Internal Harrell's C [95\\% CI]", "Internal Uno's C", f"{ext} Harrell's C [95\\% CI]", f"{ext} Uno's C"],
                         [[c[0].replace("K=", "")] + c[1:] for c in rows[1:]],
                         f"M4 performance in {COHORT_NAME[coh][0]} with $K=5$, $11$, and $20$."))
        rows = md_table(P / "results_ksens.md", f"## {coh} — 비교쌍")
        body = []
        for c in rows[1:]:
            split = "Internal" if c[1] == "internal" else ext
            body.append([tex_escape(c[0]).replace("K=", "$K=$"), split, tex_escape(c[2]), tex_escape(c[3])])
        out.append(table(["Comparison", "Evaluation", "Harrell $\\Delta$C [95\\% CI] $p$", "Uno $\\Delta$C [95\\% CI] $p$"], body,
                         f"Paired bootstrap comparison of $K=11$ against $K=5$ and $K=20$ in {COHORT_NAME[coh][0]}.", align="llcc"))
    return out


def sec_arch():
    rows = md_table(P / "results_baselines.md", "**아키텍처 수준")
    body = []
    for c in rows[1:]:
        split = "Internal" if c[1] == "internal" else "External"
        body.append([tex_escape(c[0]), split, c[2], c[3], c[4], tex_escape(c[5]), tex_escape(c[6])])
    return ["\\subsection*{Architecture-Level Comparison with PORPOISE}",
            "In this comparison PORPOISE receives this study's own inputs instead of its original ones. These are the same TCGA-PAAD cohort, folds, and 1500-gene RNA-seq panel, normalized with training-fold statistics. Any remaining difference therefore reflects the architecture rather than the input data.", "",
            table(["Comparison", "Evaluation", "$N$", "PORPOISE C", "Ours C", "Harrell $\\Delta$C [95\\% CI] $p$", "Uno $\\Delta$C [95\\% CI] $p$"], body,
                  "Architecture-level comparison of this study's models with PORPOISE trained on this study's own inputs, on TCGA-PAAD and CPTAC-PDAC.", align="llccccc")]


def sec_power():
    txt = (P / "results_power_coattn.txt").read_text(encoding="utf-8")
    out = ["\\subsection*{Statistical Power on CPTAC-PDAC}",
           "For each cohort size $N$, $N$ CPTAC-PDAC patients were resampled with replacement 400 times from the observed external predictions, which keeps the observed event rate on average. The standard error of the paired C-index difference across these resamples gives the power of a two-sided test at 0.05 by normal approximation. Power is shown for the observed difference and for a difference of 0.025.", ""]
    for block in re.split(r"\n(?=\S+ -> \S+:)", txt):
        m = re.match(r"(\S+) -> (\S+): N=\d+ events=\d+ dC=([+-][\d.]+)", block)
        if not m:
            continue
        rows = []
        for l in block.splitlines():
            r = re.match(r"\s*N=\s*(\d+) \(~(\d+) events\) SE=([\d.]+) power@obs=([\d.]+) power@0.025=([\d.]+)", l)
            if r:
                rows.append(list(r.groups()))
        out.append(table(["$N$", "Events", "SE of $\\Delta$C", f"Power at $\\Delta$C = {m.group(3)}", "Power at $\\Delta$C = +0.025"], rows,
                         f"Estimated power of the external paired comparison {m.group(1)} $\\to$ {m.group(2)}, whose observed difference on CPTAC-PDAC is {m.group(3)}."))
    return out


def main():
    uno = pd.read_csv(P / "results_uno_tdauc_models.tsv", sep="\t")
    hr = pd.read_csv(P / "results_hr_ibs_calib_models.tsv", sep="\t")
    cox = pd.read_csv(P / "results_multivariable_cox_models.tsv", sep="\t")
    parts = ["% !TEX root = ../main.tex", "%", "% Supplementary Information -- additional evaluation results",
             "% Generated by scripts/make_supplementary_results.py from paper/results_*.{md,tsv}. Do not edit by hand,",
             "% rerun the script after the analysis outputs change.", "",
             "\\section*{Supplementary Results}",
             "All results below are aggregated over five random seeds and five folds, with the best-validation and final-epoch checkpoints averaged, as in the main text. Supplementary Tables S1 to S3 are the gene panels, provided as separate files. Supplementary Tables S4 and S5 report Uno's C-index and time-dependent AUC. Supplementary Tables S6 and S7 report hazard ratios. Supplementary Tables S8 and S9 report the integrated Brier score and calibration. Supplementary Tables S10 to S14 report the multivariable Cox analysis. Supplementary Tables S15 to S18 report the sensitivity to the number of clusters. Supplementary Table S19 reports the architecture-level comparison with PORPOISE. Supplementary Tables S20 and S21 report the statistical power on CPTAC-PDAC.", ""]
    for sec in (sec_uno(uno), sec_hr(hr), sec_ibs(hr), sec_cox(cox), sec_ksens(), sec_arch(), sec_power()):
        parts += sec
    # 본문(main_new.tex)이 표 번호 S4~S21을 직접 가리키므로, 표 구성이 바뀌면 본문 번호와 위 안내 문단도 함께 고쳐야 한다
    n_tables = sum(p.count("\\begin{table}") for p in parts)
    if n_tables != 18:
        raise SystemExit(f"표가 {n_tables}개 — S4~S21(18개) 가정이 깨짐. 본문 Supplementary Table 번호를 다시 맞출 것")
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"저장: {OUT} (표 {n_tables}개, Supplementary Table S4~S21)")


if __name__ == "__main__":
    main()
