"""
2026-10-04: 리뷰 대응 재실행 배치(fold-safe, PAAD N=110 동일 코호트, BRCA 기관 단위 CV) 전체 분석.

- 모델별 pooled C-index(internal/external) + 환자 단위 bootstrap 95% CI
- 미리 정한 비교쌍 paired bootstrap delta + BH-FDR(코호트 x split 단위 family)
- 5 seeds(84,126,42,168,210), best+final 체크포인트 평균(--include-final-epoch와 동일)
- BRCA internal은 seed마다 internal test 환자가 달라 합집합(union) 앙상블

출력: paper/results_foldsafe_batch.md, paper/results_foldsafe_batch_models.tsv, _pairs.tsv
사용법: python scripts/analyze_foldsafe_batch.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from paired_bootstrap_delta import _ensemble_internal, _ensemble_external, paired_bootstrap_delta  # noqa: E402
from utils.metrics import compute_survival_metrics  # noqa: E402

SEEDS = [84, 126, 42, 168, 210]
N_FOLDS = 5
N_BOOT = 2000
PRED = _ROOT / ".logs"

PAAD = {
    "M1": "M1_uni2native_SS_CLUSTERPOOL_FS_COHN110_NLLSURV4_NLLCOX1",
    "M2": "M2_uni2native_SS_STG_R_CLUSTERPOOL_FS_COHN110_CLR100_NLLSURV4_NLLCOX1",
    "M3": "PMA_uni2native_PDACCONS1500_CNV_SS_CLUSTERPOOL_FS_COHN110_NOCLINICAL_NLLSURV4_NLLCOX1",
    "M4": "PMA_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_CLUSTERPOOL_FS_COHN110_COX_ADD_CLR100_NLLSURV4_NLLCOX1",
    "M5": "M5_STG_R_RAWLIN_NLLSURV4_NLLCOX1_FS_COHN110",
    "M6": "M6_PDACCONS1500_CNV_NLLSURV4_NLLCOX1_FS_COHN110",
    "M7": "M7_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_CLR100_NLLSURV4_NLLCOX1_FS_COHN110",
    "M3-SA": "PMA_uni2native_PDACCONS1500_CNV_SS_SELFATTNFUSION_CLUSTERPOOL_FS_COHN110_NOCLINICAL_NLLSURV4_NLLCOX1",
    "M4-SA": "PMA_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_SELFATTNFUSION_CLUSTERPOOL_FS_COHN110_COX_ADD_CLR100_NLLSURV4_NLLCOX1",
    "M3-noCNV": "PMA_uni2native_PDACCONS1500_SS_CLUSTERPOOL_FS_COHN110_NOCLINICAL_NLLSURV4_NLLCOX1",
    "M4-noCNV": "PMA_uni2native_PDACCONS1500_SS_STG_R_MUT_CLUSTERPOOL_FS_COHN110_COX_ADD_CLR100_NLLSURV4_NLLCOX1",
    "M6-noCNV": "M6_PDACCONS1500_NLLSURV4_NLLCOX1_FS_COHN110",
    "M7-noCNV": "M7_PDACCONS1500_STG_R_MUT_COX_ADD_CLR100_NLLSURV4_NLLCOX1_FS_COHN110",
    "M4-noMUT": "PMA_uni2native_PDACCONS1500_CNV_SS_STG_R_CLUSTERPOOL_FS_COHN110_COX_ADD_CLR100_NLLSURV4_NLLCOX1",
}
BRCA = {
    "M1": "BRCA_M1_CLUSTERPOOL_NLLSURV4_NLLCOX1_INSTCV_FS",
    "M2": "BRCA_M2_STG_CLR100_CLUSTERPOOL_NLLSURV4_NLLCOX1_INSTCV_FS",
    "M3": "BRCA_PMA_CONS882_NOCLINICAL_SS_CLUSTERPOOL_NLLSURV4_NLLCOX1_INSTCV_FS",
    "M4": "BRCA_PMA_CONS882_STG_SS_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1_INSTCV_FS",
    "M5": "BRCA_M5_STG_NLLSURV4_NLLCOX1_INSTCV_FS",
    "M6": "BRCA_M6_CONS882_NLLSURV4_NLLCOX1_INSTCV_FS",
    "M7": "BRCA_M7_CONS882_STG_NLLSURV4_NLLCOX1_CLR100_INSTCV_FS",
    "M3-SA": "BRCA_PMA_CONS882_NOCLINICAL_SELFATTNFUSION_SS_CLUSTERPOOL_NLLSURV4_NLLCOX1_INSTCV_FS",
    "M4-SA": "BRCA_PMA_CONS882_SELFATTNFUSION_STG_SS_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1_INSTCV_FS",
}
MODALITY_PAIRS = [
    ("WSI", "M6", "M3"), ("WSI", "M5", "M2"), ("WSI", "M7", "M4"),
    ("RNA-seq+CNV", "M1", "M3"), ("RNA-seq+CNV", "M5", "M7"), ("RNA-seq+CNV", "M2", "M4"),
    ("Clinical", "M1", "M2"), ("Clinical", "M6", "M7"), ("Clinical", "M3", "M4"),
    ("co-attention", "M3-SA", "M3"), ("co-attention", "M4-SA", "M4"),
]
PAAD_EXTRA_PAIRS = [
    ("CNV", "M3-noCNV", "M3"), ("CNV", "M4-noCNV", "M4"), ("CNV", "M6-noCNV", "M6"), ("CNV", "M7-noCNV", "M7"),
    ("RNA-seq only", "M1", "M3-noCNV"), ("RNA-seq only", "M5", "M7-noCNV"),
    ("mutation", "M4-noMUT", "M4"),
]
COHORTS = {
    "PAAD": dict(models=PAAD, internal="tcga", external="cptac", union=False,
                 pairs=MODALITY_PAIRS + PAAD_EXTRA_PAIRS),
    "BRCA": dict(models=BRCA, internal="brca", external="brca", union=True, pairs=MODALITY_PAIRS),
}


def load(cohort, split, model):
    c = COHORTS[cohort]
    if split == "internal":
        return _ensemble_internal(PRED / "kfold_preds", c["internal"], model, SEEDS, N_FOLDS,
                                  include_final_epoch=True, union_seeds=c["union"])
    return _ensemble_external(PRED / "external_preds", c["external"], model, SEEDS, N_FOLDS,
                              include_final_epoch=True)


def boot_ci(r, t, e, n=N_BOOT, seed=0):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, len(r), len(r))
        c = compute_survival_metrics(r[idx], t[idx], e[idx])["c_index"]
        if not np.isnan(c):
            vals.append(c)
    return np.percentile(vals, [2.5, 97.5])


def bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def main():
    cache = {}
    model_rows, pair_rows = [], []
    for cohort, c in COHORTS.items():
        for name, tag in c["models"].items():
            row = {"cohort": cohort, "model": name}
            for split in ("internal", "external"):
                ids, r, t, e = load(cohort, split, tag)
                cache[(cohort, split, name)] = (ids, r, t, e)
                m = compute_survival_metrics(r, t, e)
                lo, hi = boot_ci(r, t, e)
                row.update({f"{split}_n": len(ids), f"{split}_events": int(e.sum()),
                            f"{split}_c": m["c_index"], f"{split}_lo": lo, f"{split}_hi": hi,
                            f"{split}_logrank_p": m["log_rank_p"]})
            model_rows.append(row)
            print(f"{cohort} {name:9s} int {row['internal_c']:.4f} ext {row['external_c']:.4f}", flush=True)
        for group, a, b in c["pairs"]:
            for split in ("internal", "external"):
                A = cache[(cohort, split, a)]
                B = cache[(cohort, split, b)]
                res = paired_bootstrap_delta(*A, *B, n_bootstrap=N_BOOT)
                pair_rows.append({"cohort": cohort, "group": group, "pair": f"{a} -> {b}", "split": split,
                                  "n": res["n_patients"], "delta": res["point_delta"],
                                  "lo": res["delta_ci_lo"], "hi": res["delta_ci_hi"], "p": res["p_value"]})

    models = pd.DataFrame(model_rows)
    pairs = pd.DataFrame(pair_rows)
    pairs["q"] = np.nan
    for (cohort, split), idx in pairs.groupby(["cohort", "split"]).groups.items():
        pairs.loc[idx, "q"] = bh(pairs.loc[idx, "p"])

    out = _ROOT / "paper"
    models.to_csv(out / "results_foldsafe_batch_models.tsv", sep="\t", index=False)
    pairs.to_csv(out / "results_foldsafe_batch_pairs.tsv", sep="\t", index=False)

    def fmt_c(r, s):
        return f"{r[f'{s}_c']:.4f} [{r[f'{s}_lo']:.4f}, {r[f'{s}_hi']:.4f}]"

    def fmt_p(p):
        return "<0.0001" if p < 1e-4 else f"{p:.4f}"

    lines = ["# fold-safe 재실행 배치 결과 (2026-10-04)", "",
             "5 seeds x 5 folds, best+final 체크포인트 평균, 환자 단위 bootstrap 2000회. "
             "PAAD: 전 모델 N=110(TCGA) / 136(CPTAC) 동일 코호트. BRCA: 기관 단위 CV(external = 학습에 안 쓴 기관, "
             "모든 환자 1회씩), internal은 seed 합집합 앙상블. q = BH-FDR(코호트 x split별).", ""]
    for cohort in COHORTS:
        lines += [f"## {cohort} — 모델별 C-index", "",
                  "| model | internal C [95% CI] | logrank p | external C [95% CI] | logrank p |",
                  "|---|---|---|---|---|"]
        for _, r in models[models.cohort == cohort].iterrows():
            lines.append(f"| {r.model} | {fmt_c(r, 'internal')} | {fmt_p(r.internal_logrank_p)} | "
                         f"{fmt_c(r, 'external')} | {fmt_p(r.external_logrank_p)} |")
        r0 = models[models.cohort == cohort].iloc[0]
        lines += ["", f"N: internal {r0.internal_n} ({r0.internal_events} events), "
                      f"external {r0.external_n} ({r0.external_events} events)", "",
                  f"## {cohort} — 비교쌍 (A -> B, delta = B - A)", "",
                  "| 그룹 | 비교 | internal ΔC [95% CI] p (q) | external ΔC [95% CI] p (q) |", "|---|---|---|---|"]
        sub = pairs[pairs.cohort == cohort]
        for (group, pair), g in sub.groupby(["group", "pair"], sort=False):
            cells = []
            for split in ("internal", "external"):
                x = g[g.split == split].iloc[0]
                star = "**" if (x.lo > 0 or x.hi < 0) else ""
                cells.append(f"{star}{x.delta:+.4f} [{x.lo:+.4f}, {x.hi:+.4f}]{star} {x.p:.3f} ({x.q:.3f})")
            lines.append(f"| {group} | {pair} | {cells[0]} | {cells[1]} |")
        lines.append("")
    (out / "results_foldsafe_batch.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
