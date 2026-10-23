"""
2026-10-20: GPT 사전 검토 P1 — gap 표의 "Molecular gain in the same model" 열 분해.

그 열은 C(XA 모델) − C(molecular 없는 모델)인데, molecular 없는 쪽(M1, M5)은 self-attention이고
XA 모델(M4, M7)은 cross-attention이라 그 차이 안에 cross-attention 효과가 이미 들어 있다.
    C(XA) − C(base) = [C(SA) − C(base)]  +  [C(XA) − C(SA)]
                       입력 추가(fusion 고정)    cross-attention 효과
두 항을 같은 환자로 paired bootstrap한다(외부·내부 각각). 재학습 없음, 기존 환자별 예측만 사용.
PAAD의 M7-SA − M5에는 mutation flag 4개 추가도 섞여 있다(M5에는 없음).

분석 산출물의 옛 번호: 원고 M4 = 옛 M3, 원고 M5 = 옛 M2, 원고 M7 = 옛 M4.
사용법: python -m scripts.analyze_mol_gain_decomposition [--n-boot 2000] -> paper/results_mol_gain_decomposition.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from analyze_coattn_gap import arrays, harrell, load  # noqa: E402

COMPARISONS = [  # (cohort, 원고 이름, base, SA, XA) — 분석 ID
    ("PAAD", "M4", "M1", "M3-SA", "M3"),
    ("PAAD", "M7", "M2", "M4-SA", "M4"),
    ("BRCA", "M4", "M1", "M3-SA", "M3"),
    ("BRCA", "M7", "M2", "M4-SA", "M4"),
]


def terms(p, ids):
    c = {k: harrell(*arrays(p[k], ids)) for k in ("base", "SA", "XA")}
    return np.array([c["XA"] - c["base"], c["SA"] - c["base"], c["XA"] - c["SA"]])


def summarize(point, boots):
    out = []
    for j in range(3):
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
        p = min(1.0, 2 * min((boots[:, j] <= 0).mean(), (boots[:, j] >= 0).mean()))
        out.append((point[j], lo, hi, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.RandomState(0)
    rows = []
    for cohort, name, base, sa, xa in COMPARISONS:
        for split in ("external", "internal"):
            p = {k: load(cohort, split, m) for k, m in (("base", base), ("SA", sa), ("XA", xa))}
            ids = sorted(set(p["base"]) & set(p["SA"]) & set(p["XA"]))
            point = terms(p, ids)
            boots = np.array([terms(p, [ids[i] for i in rng.randint(0, len(ids), len(ids))]) for _ in range(args.n_boot)])
            rows.append((cohort, name, split, len(ids), summarize(point, boots)))
            s = summarize(point, boots)
            print(f"{cohort} {name} {split}: total {s[0][0]:+.4f} | SA-based input {s[1][0]:+.4f} [{s[1][1]:+.4f}, {s[1][2]:+.4f}] p={s[1][3]:.3f} "
                  f"| XA {s[2][0]:+.4f} [{s[2][1]:+.4f}, {s[2][2]:+.4f}] p={s[2][3]:.3f}", flush=True)

    f = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}] | {t[3]:.3f}"
    L = ["# Molecular gain 분해 — 입력 추가 vs cross-attention", "",
         "C(XA) − C(base) = [C(SA) − C(base)] + [C(XA) − C(SA)]. base는 molecular 없는 모델(M4는 M1, M7은 M5). "
         f"Harrell's C, 같은 환자 paired bootstrap {args.n_boot}회. 번호는 원고 번호. PAAD M7의 입력 추가 항에는 mutation flag 추가가 섞여 있음.", "",
         "| Cohort | Model | Split | N | 현재 열: C(XA) − C(base) | p | 입력 추가, SA 기준: C(SA) − C(base) [95% CI] | p | Cross-attention: C(XA) − C(SA) [95% CI] | p |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for cohort, name, split, n, s in rows:
        L.append(f"| {cohort} | {name} | {split} | {n} | {f(s[0])} | {f(s[1])} | {f(s[2])} |")
    out = _ROOT / "paper" / "results_mol_gain_decomposition.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
