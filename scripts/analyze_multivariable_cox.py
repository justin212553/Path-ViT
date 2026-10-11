"""
2026-10-10: SciRep 리뷰 Q2 — "모델 risk score가 병기·나이·성별 보정 후에도 독립적 예후 인자인가"
다변량 Cox. 재학습 없이 저장된 앙상블 risk(analyze_foldsafe_batch.load, best+final 평균)를 쓴다.

공변량
- risk: internal 평균/표준편차로 표준화(1 SD당 HR). external도 같은 변환
- age: 10세당
- PAAD: sex(male), N+(N1/N2 vs N0), T≥3. TCGA(AJCC 7판)와 CPTAC(8판 분포)의 병기 기준판이 달라
  T 정의가 다르므로 판에 덜 민감한 N+를 중심으로, T는 이분형으로만
- BRCA: T, N 순서형(models.clinical_encoder.encode_stage_value), M1(확인된 M1만 1, MX는 0). 전원 여성이라 sex 제외
- 결측(TX/NX/MX 포함)은 complete-case로 제외, 분석 n을 보고

모델: M4(병기를 입력으로 받음 — risk와 stage가 겹침), M3(WSI+RNA, 임상 입력 없음 — 병기와 독립성의
가장 깨끗한 검정), 참고로 M7, M6. 추가 기여는 임상 공변량만의 Cox 대비 risk를 더한 모형의
likelihood-ratio test(자유도 1).

출력: paper/results_multivariable_cox.md, _models.tsv
사용법: python scripts/analyze_multivariable_cox.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from analyze_foldsafe_batch import COHORTS, load  # noqa: E402
from models.clinical_encoder import encode_stage_value  # noqa: E402

MODELS = ["M4", "M3", "M7", "M6"]
CLIN = {"PAAD": {"internal": "data/clinical_tcga.csv", "external": "data/clinical_cptac.csv"},
        "BRCA": {"internal": "data/brca_clinical.csv", "external": "data/brca_clinical.csv"}}
LABELS = {"risk": "model risk (per SD)", "age10": "age (per 10 y)", "male": "sex (male)",
          "n_pos": "N+ (vs N0)", "t3": "T≥3", "t_ord": "T (ordinal)", "n_ord": "N (ordinal)", "m1": "M1"}


def covariates(cohort, path):
    c = pd.read_csv(_ROOT / path).set_index("case_id")
    out = pd.DataFrame(index=c.index)
    out["age10"] = c["age_years"] / 10.0
    t = c["ajcc_t"].map(lambda v: encode_stage_value("ajcc_t", v))
    n = c["ajcc_n"].map(lambda v: encode_stage_value("ajcc_n", v))
    if cohort == "PAAD":
        out["male"] = (c["sex"] == "male").astype(float)
        out["n_pos"] = n.map(lambda v: np.nan if v is None or pd.isna(v) else float(v > 0))
        out["t3"] = t.map(lambda v: np.nan if v is None or pd.isna(v) else float(v >= 3))
    else:
        out["t_ord"] = t.astype(float)
        out["n_ord"] = n.astype(float)
        m = c["ajcc_m"].map(lambda v: encode_stage_value("ajcc_m", v))
        # MX(미평가, 161명)를 결측으로 빼면 16%가 날아간다 — 확인된 M1만 1, M0/MX는 0
        out["m1"] = m.map(lambda v: 0.0 if v is None or pd.isna(v) else float(v > 0))
    return out


def fit(df, cols):
    cph = CoxPHFitter().fit(df[cols + ["T", "E"]], "T", "E")
    return cph


def main():
    rows, tables = [], {}
    for cohort, c in COHORTS.items():
        for name in MODELS:
            tag = c["models"][name]
            _, r_i, _, _ = load(cohort, "internal", tag)
            mu, sd = r_i.mean(), r_i.std()
            for split in ("internal", "external"):
                ids, r, t, e = load(cohort, split, tag)
                cov = covariates(cohort, CLIN[cohort][split])
                df = pd.DataFrame({"risk": (r - mu) / sd, "T": np.clip(np.asarray(t, float), 0, None),
                                   "E": np.asarray(e, int)}, index=ids).join(cov, how="left")
                clin_cols = [k for k in cov.columns]
                n_all = len(df)
                df = df.dropna()
                full = fit(df, ["risk"] + clin_cols)
                base = fit(df, clin_cols)
                lr = 2 * (full.log_likelihood_ - base.log_likelihood_)
                s = full.summary
                row = {"cohort": cohort, "model": name, "split": split, "n": len(df), "n_excluded": n_all - len(df),
                       "events": int(df.E.sum()),
                       "hr": float(s.loc["risk", "exp(coef)"]), "lo": float(s.loc["risk", "exp(coef) lower 95%"]),
                       "hi": float(s.loc["risk", "exp(coef) upper 95%"]), "p": float(s.loc["risk", "p"]),
                       "lr_p": float(chi2.sf(lr, 1)),
                       "c_clin": float(base.concordance_index_), "c_full": float(full.concordance_index_)}
                rows.append(row)
                tables[(cohort, name, split)] = s
                print(f"{cohort} {name} {split}: n={len(df)} HR={row['hr']:.2f} [{row['lo']:.2f},{row['hi']:.2f}] "
                      f"p={row['p']:.4f} LR p={row['lr_p']:.4f}", flush=True)

    m = pd.DataFrame(rows)
    out = _ROOT / "paper"
    m.to_csv(out / "results_multivariable_cox_models.tsv", sep="\t", index=False)

    def fp(p):
        return "<0.0001" if p < 1e-4 else f"{p:.4f}"

    L = ["# 다변량 Cox (fold-safe 배치, 2026-10-10)", "",
         "모델 risk(internal 기준 1 SD당) + 임상 공변량. PAAD: 나이(10세당), 성별, N+, T≥3. "
         "BRCA: 나이, T·N 순서형, M1(전원 여성이라 성별 제외). 결측·판정불가 병기는 complete-case 제외. "
         "LR p = 임상 공변량만의 Cox 대비 risk 추가의 likelihood-ratio test. C(clin)/C(full) = 각 Cox 모형의 "
         "적합 데이터 C-index(같은 데이터로 적합해 낙관적, 상대 비교용). "
         "M4는 병기를 입력으로 받으므로 risk와 병기가 겹친다 — 병기와의 독립성은 임상 입력이 없는 M3/M6가 더 깨끗한 검정.", "",
         "주의: PAAD는 TCGA(AJCC 7판)와 CPTAC(8판 분포)의 병기 기준판이 달라 T 정의가 다름 — 판에 덜 민감한 N+ 중심.", ""]
    for cohort in COHORTS:
        x = m[m.cohort == cohort]
        L += [f"## {cohort} — 보정 후 model risk HR", "",
              "| model | split | n (제외) | events | adjusted HR per SD [95% CI] | p | LR p (risk 추가) | C clin → full |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in x.iterrows():
            L.append(f"| {r.model} | {r.split} | {r.n} ({r.n_excluded}) | {r.events} | "
                     f"{r.hr:.2f} [{r.lo:.2f}, {r.hi:.2f}] | {fp(r.p)} | {fp(r.lr_p)} | {r.c_clin:.3f} → {r.c_full:.3f} |")
        L.append("")
        for name in ("M4", "M3"):
            L += [f"### {cohort} {name} — 전체 공변량", "",
                  "| 변수 | internal HR [95% CI] p | external HR [95% CI] p |", "|---|---|---|"]
            si, se = tables[(cohort, name, "internal")], tables[(cohort, name, "external")]
            for v in si.index:
                cells = [f"{s.loc[v, 'exp(coef)']:.2f} [{s.loc[v, 'exp(coef) lower 95%']:.2f}, "
                         f"{s.loc[v, 'exp(coef) upper 95%']:.2f}] {fp(s.loc[v, 'p'])}" for s in (si, se)]
                L.append(f"| {LABELS.get(v, v)} | {cells[0]} | {cells[1]} |")
            L.append("")
    (out / "results_multivariable_cox.md").write_text("\n".join(L), encoding="utf-8")
    print("written paper/results_multivariable_cox.md")


if __name__ == "__main__":
    main()
