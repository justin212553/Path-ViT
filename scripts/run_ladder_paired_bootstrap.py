"""
M1~M7 결과표의 "모달리티 추가 효과"를 9쌍(RNA 추가 3쌍 x Clinic 추가 3쌍 x WSI 추가 3쌍 —
정확히는 3종류 추가 효과 x 각 3개 baseline)으로 paired bootstrap delta 비교한다.
scripts/paired_bootstrap_delta.py를 모듈로 불러와 internal/external 각각 9쌍씩(총 18회) 돌리고
markdown 표로 정리해서 출력한다. 2026-08-21, 외부 피드백(paired bootstrap on delta) 반영.

사용법: python scripts/run_ladder_paired_bootstrap.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paired_bootstrap_delta import _ensemble_internal, _ensemble_external, paired_bootstrap_delta

FINAL_MODELS = {
    "M1": "M1_POOL_uni2native_SS_DISP",
    "M2": "M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD_SELFATTN",
    "M3": "PMA_uni2native_INT1500_SS_AUX_NOCLINICAL_DISP",
    "M4": "PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD",
    "M5": "M5_STG_R_RAWLIN",
    "M6": "M6_INT1500",
    "M7": "M7_INT1500_STG_R_COX_ADD",
}

# (효과, A(baseline), B(A+효과)) — 3효과 x 3개 baseline = 9쌍
PAIRS = [
    ("+RNA",    "M1", "M3"),
    ("+RNA",    "M5", "M7"),
    ("+RNA",    "M2", "M4"),
    ("+Clinic", "M1", "M2"),
    ("+Clinic", "M6", "M7"),
    ("+Clinic", "M3", "M4"),
    ("+WSI",    "M5", "M2"),
    ("+WSI",    "M6", "M3"),
    ("+WSI",    "M7", "M4"),
]

PRED_ROOT = _ROOT.parent / "paper" / "final_preds_snapshot"
SEEDS = [84, 126]
N_FOLDS = 5
N_BOOTSTRAP = 2000


def _run_split(split: str, dataset: str):
    ensemble_fn = _ensemble_internal if split == "internal" else _ensemble_external
    pred_dir = PRED_ROOT / ("kfold_preds" if split == "internal" else "external_preds")

    cache = {}
    def get(tag):
        if tag not in cache:
            model = FINAL_MODELS[tag]
            cache[tag] = ensemble_fn(pred_dir, dataset, model, SEEDS, N_FOLDS)
        return cache[tag]

    rows = []
    print(f"\n{'=' * 70}\n{split.upper()} ({dataset})\n{'=' * 70}")
    for effect, tag_a, tag_b in PAIRS:
        cases_a, risks_a, times_a, events_a = get(tag_a)
        cases_b, risks_b, times_b, events_b = get(tag_b)
        print(f"\n--- {effect}: {tag_a} -> {tag_b} ---")
        r = paired_bootstrap_delta(cases_a, risks_a, times_a, events_a,
                                    cases_b, risks_b, times_b, events_b,
                                    n_bootstrap=N_BOOTSTRAP)
        sig = "**유의**" if r["significant"] else "비유의"
        print(f"  N(paired)={r['n_patients']} | C({tag_a})={r['point_a']:.4f} | C({tag_b})={r['point_b']:.4f} | "
              f"delta={r['point_delta']:+.4f}")
        print(f"  95% CI=[{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}] | p={r['p_value']:.4f} -> {sig}")
        rows.append((effect, tag_a, tag_b, r))
    return rows


def main():
    internal_rows = _run_split("internal", "tcga")
    external_rows = _run_split("external", "cptac")

    print(f"\n\n{'=' * 70}\nMARKDOWN SUMMARY\n{'=' * 70}\n")
    print("### Internal\n")
    print("| 효과 | A | B | N(paired) | C(A) | C(B) | delta(B-A) | 95% CI | p | 판정 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for effect, tag_a, tag_b, r in internal_rows:
        sig = "유의" if r["significant"] else "비유의"
        print(f"| {effect} | {tag_a} | {tag_b} | {r['n_patients']} | {r['point_a']:.4f} | {r['point_b']:.4f} | "
              f"{r['point_delta']:+.4f} | [{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}] | "
              f"{r['p_value']:.4f} | {sig} |")

    print("\n### External\n")
    print("| 효과 | A | B | N(paired) | C(A) | C(B) | delta(B-A) | 95% CI | p | 판정 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for effect, tag_a, tag_b, r in external_rows:
        sig = "유의" if r["significant"] else "비유의"
        print(f"| {effect} | {tag_a} | {tag_b} | {r['n_patients']} | {r['point_a']:.4f} | {r['point_b']:.4f} | "
              f"{r['point_delta']:+.4f} | [{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}] | "
              f"{r['p_value']:.4f} | {sig} |")


if __name__ == "__main__":
    main()
