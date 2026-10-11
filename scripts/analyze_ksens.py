"""
2026-10-10: K 민감도 재실험(fold-safe, 5 seeds x 5 folds) 분석 — M4의 ClusterPool K=5/11/20 비교.
K=11은 기존 fold-safe 배치의 M4 그대로(analyze_foldsafe_batch.py의 PAAD/BRCA["M4"]).

- 모델별 Harrell's C, Uno's C(+ 환자 단위 bootstrap 95% CI)
- K=5 -> K=11, K=20 -> K=11 paired bootstrap ΔC(Harrell, Uno), p는 paired_bootstrap_delta와 같은 정의
- 앙상블/코호트 정의, best+final 평균, BRCA internal seed 합집합은 analyze_foldsafe_batch.py와 동일

출력: paper/results_ksens.md
사용법: python scripts/analyze_ksens.py [--n-boot 2000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from analyze_foldsafe_batch import PAAD, BRCA, load  # noqa: E402
from analyze_uno_tdauc import TAU, harrell_c, uno_c, align  # noqa: E402

MODELS = {
    "PAAD": {
        "K=5": "PMA_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_CLUSTERPOOL_FS_K5_COHN110_COX_ADD_CLR100_NLLSURV4_NLLCOX1",
        "K=11": PAAD["M4"],
        "K=20": "PMA_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_CLUSTERPOOL_FS_K20_COHN110_COX_ADD_CLR100_NLLSURV4_NLLCOX1",
    },
    "BRCA": {
        "K=5": "BRCA_PMA_CONS882_STG_SS_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1_INSTCV_FS_K5",
        "K=11": BRCA["M4"],
        "K=20": "BRCA_PMA_CONS882_STG_SS_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1_INSTCV_FS_K20",
    },
}
PAIRS = [("K=5", "K=11"), ("K=20", "K=11")]


def metric(kind, cohort):
    if kind == "harrell":
        return lambda r, t, e: (lambda n, d: n / d)(*harrell_c(r, t, e))
    return lambda r, t, e: uno_c(r, t, e, TAU[cohort])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    lines = ["# K 민감도 (fold-safe 재실험, 2026-10-10)", "",
             "M4, 5 seeds x 5 folds, best+final 평균. fold-safe라 centroid는 fold마다 train 슬라이드로 해당 K에 "
             f"재적합. 환자 단위 bootstrap {args.n_boot}회. Uno's C 절단 tau: PAAD 3년, BRCA 5년. "
             "PAAD N=110/136 동일 코호트, BRCA 기관 단위 CV.", ""]
    for cohort, models in MODELS.items():
        data = {}
        for k, tag in models.items():
            for split in ("internal", "external"):
                ids, r, t, e = load(cohort, split, tag)
                data[(k, split)] = (ids, r, np.asarray(t, float), np.asarray(e, int))

        lines += [f"## {cohort} — 모델별", "",
                  "| K | internal Harrell [95% CI] | internal Uno | external Harrell [95% CI] | external Uno |",
                  "|---|---|---|---|---|"]
        rng = np.random.RandomState(0)
        for k in models:
            cells = []
            for split in ("internal", "external"):
                _, r, t, e = data[(k, split)]
                h = metric("harrell", cohort)(r, t, e)
                boots = []
                for _ in range(args.n_boot):
                    ix = rng.randint(0, len(t), len(t))
                    if e[ix].sum():
                        boots.append(metric("harrell", cohort)(r[ix], t[ix], e[ix]))
                lo, hi = np.percentile(boots, [2.5, 97.5])
                u = metric("uno", cohort)(r, t, e)
                cells += [f"{h:.4f} [{lo:.4f}, {hi:.4f}]", f"{u:.4f}"]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")

        lines += [f"## {cohort} — 비교쌍 (A -> B, Δ = B − A)", "",
                  "| 비교 | split | Harrell Δ [95% CI] p | Uno Δ [95% CI] p |", "|---|---|---|---|"]
        for a, b in PAIRS:
            for split in ("internal", "external"):
                A, B = data[(a, split)], data[(b, split)]
                ra, rb, t, e = align(A, B)
                cells = []
                for kind in ("harrell", "uno"):
                    f = metric(kind, cohort)
                    point = f(rb, t, e) - f(ra, t, e)
                    ds = []
                    for _ in range(args.n_boot):
                        ix = rng.randint(0, len(t), len(t))
                        if e[ix].sum():
                            ds.append(f(rb[ix], t[ix], e[ix]) - f(ra[ix], t[ix], e[ix]))
                    ds = np.asarray(ds)
                    lo, hi = np.percentile(ds, [2.5, 97.5])
                    p = min(2 * min(np.mean(ds <= 0), np.mean(ds >= 0)), 1.0)
                    cells.append(f"{point:+.4f} [{lo:+.4f}, {hi:+.4f}] {p:.3f}")
                lines.append(f"| {a} -> {b} | {split} | {cells[0]} | {cells[1]} |")
            print(f"{cohort} {a}->{b} done", flush=True)
        lines.append("")
    (_ROOT / "paper" / "results_ksens.md").write_text("\n".join(lines), encoding="utf-8")
    print("written paper/results_ksens.md")


if __name__ == "__main__":
    main()
