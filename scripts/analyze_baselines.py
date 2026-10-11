"""
2026-10-10: 새 프로토콜(fold-safe, 5 seeds) 기준 published baseline 비교(사용자 결정 C안).
- PAAD PORPOISE: 공식 코드·레시피·WSI 스펙, 우리와 같은 코호트(110/136)·같은 seed별 fold·같은 RNA
  (pdac_consistency_1500 raw log2, PORPOISE 자체 fold-train scaler) — scripts/prepare_porpoise_paad_n110.py
- BRCA MMP: 공식 코드·공식 Hallmark RNA(아키텍처를 정의하는 pathway 패널이라 유지), 우리 기관 단위 CV split
예측은 scripts/export_baseline_preds.py가 .logs에 써둔 CSV.

- 우리 모델(M4, M3, M7, M4-SA): analyze_foldsafe_batch.load (best+final 평균)
- baseline: 마지막 epoch 하나뿐이라 include_final_epoch=False. MMP internal은 seed 합집합(기관 단위 CV)
- 비교는 두 모델 모두 예측이 있는 환자 교집합에서 paired bootstrap(Harrell, Uno)
- 공정성 메모: PORPOISE·MMP는 임상 입력이 없다 → 같은 입력 조합인 M3(WSI+RNA)와의 비교가 구조 비교,
  M4와의 비교는 "우리 전체 모델 vs 공개 모델" 비교

출력: paper/results_baselines.md
사용법: python scripts/analyze_baselines.py [--seeds 84,126,42,168,210] [--n-boot 2000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from analyze_foldsafe_batch import PAAD, BRCA, PRED, load  # noqa: E402
from analyze_uno_tdauc import TAU, harrell_c, uno_c, align  # noqa: E402
from paired_bootstrap_delta import _ensemble_internal, _ensemble_external  # noqa: E402

BASELINES = {
    "PAAD": dict(name="PORPOISE", tag="PORPOISE_N110_MMF", internal="tcga", external="cptac", union=False),
    "BRCA": dict(name="MMP", tag="MMP_INSTCV", internal="brca", external="brca", union=True),
}
OURS = {"PAAD": PAAD, "BRCA": BRCA}
# PORPOISE·MMP는 임상 입력이 없다 — 같은 입력 조합 비교: PAAD M3-noCNV(WSI+RNA, PORPOISE와 RNA 입력까지 동일), BRCA M3
COMPARE = {"PAAD": ["M3-noCNV", "M3", "M4", "M4-SA", "M7"], "BRCA": ["M3", "M4", "M4-SA", "M7"]}


def load_baseline(cohort, split, seeds):
    b = BASELINES[cohort]
    if split == "internal":
        ids, r, t, e = _ensemble_internal(PRED / "kfold_preds", b["internal"], b["tag"], seeds, 5,
                                          include_final_epoch=False, union_seeds=b["union"])
    else:
        ids, r, t, e = _ensemble_external(PRED / "external_preds", b["external"], b["tag"], seeds, 5,
                                          include_final_epoch=False)
    return ids, r, np.asarray(t, float), np.asarray(e, int)


def cidx(kind, cohort):
    if kind == "harrell":
        return lambda r, t, e: (lambda n, d: n / d)(*harrell_c(r, t, e))
    return lambda r, t, e: uno_c(r, t, e, TAU[cohort])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="84,126,42,168,210")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    L = ["# Published baseline 비교 (새 프로토콜, 2026-10-10)", "",
         f"baseline seeds={seeds}, 우리 모델 5 seeds x 5 folds(best+final 평균). 환자 교집합 paired bootstrap "
         f"{args.n_boot}회. Uno tau PAAD 3년, BRCA 5년. PORPOISE·MMP는 임상 입력이 없어 M3(WSI+RNA)가 같은 입력 조합.",
         "PAAD는 코호트·fold·RNA 입력이 우리 모델과 같아 아키텍처 차이만 비교됨(C안).", ""]
    for cohort, b in BASELINES.items():
        L += [f"## {cohort} — {b['name']} -> 우리 모델 (Δ = 우리 − {b['name']})", "",
              "| 비교 | split | n | baseline Harrell | 우리 Harrell | Harrell Δ [95% CI] p | Uno Δ [95% CI] p |",
              "|---|---|---|---|---|---|---|"]
        rng = np.random.RandomState(0)
        for split in ("internal", "external"):
            try:
                B = load_baseline(cohort, split, seeds)
            except (FileNotFoundError, ValueError) as ex:
                L.append(f"| — | {split} | — | 예측 없음: {ex} | | | |")
                continue
            for name in COMPARE[cohort]:
                O = load(cohort, split, OURS[cohort][name])
                ra, rb, t, e = align(B, (O[0], O[1], O[2], O[3]))
                t = t.astype(float)
                cells, base_h, our_h = [], None, None
                for kind in ("harrell", "uno"):
                    f = cidx(kind, cohort)
                    a0, b0 = f(ra, t, e), f(rb, t, e)
                    if kind == "harrell":
                        base_h, our_h = a0, b0
                    ds = []
                    for _ in range(args.n_boot):
                        ix = rng.randint(0, len(t), len(t))
                        if e[ix].sum():
                            ds.append(f(rb[ix], t[ix], e[ix]) - f(ra[ix], t[ix], e[ix]))
                    ds = np.asarray(ds)
                    lo, hi = np.percentile(ds, [2.5, 97.5])
                    p = min(2 * min(np.mean(ds <= 0), np.mean(ds >= 0)), 1.0)
                    star = "**" if lo > 0 or hi < 0 else ""
                    cells.append(f"{star}{b0 - a0:+.4f} [{lo:+.4f}, {hi:+.4f}]{star} {p:.3f}")
                L.append(f"| {b['name']} -> {name} | {split} | {len(t)} | {base_h:.4f} | {our_h:.4f} | {cells[0]} | {cells[1]} |")
                print(f"{cohort} {split} {b['name']}->{name} n={len(t)} {cells[0]}", flush=True)
        L.append("")
    (_ROOT / "paper" / "results_baselines.md").write_text("\n".join(L), encoding="utf-8")
    print("written paper/results_baselines.md")


if __name__ == "__main__":
    main()
