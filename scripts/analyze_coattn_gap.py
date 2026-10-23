"""
2026-10-19: RNA-guided cross-attention이 internal-external 격차를 줄이는지 직접 본다.

각 모델의 격차 = external C - internal C (BRCA는 held-out institution C - internal C).
cross-attention 효과 = 격차(cross-attention 모델) - 격차(self-attention 모델) = Δexternal - Δinternal.
양수면 cross-attention이 격차를 줄인(외부 쪽으로 끌어올린) 것.

bootstrap: PAAD는 internal(TCGA)과 external(CPTAC)이 서로 다른 환자라 따로 resample한다.
BRCA는 같은 환자가 internal과 held-out 양쪽에 예측을 가지므로 환자 단위로 함께 resample한다
(한 resample에서 뽑힌 환자 중 internal 예측이 있는 환자만 internal C에 쓰인다).
"RNA-seq가 그 모델에 없던 예후 정보를 주는가"의 지표로 같은 표에 RNA-seq 추가 쌍(M1->M3, M2->M4)의
external ΔC를 나란히 둔다.

사용법: python -m scripts.analyze_coattn_gap [--n-boot 2000]   -> paper/results_coattn_gap.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import analyze_foldsafe_batch as afb  # noqa: E402

COMPARISONS = [  # (cohort, self-attention model, cross-attention model, RNA-seq 추가 쌍)
    ("PAAD", "M3-SA", "M3", ("M1", "M3")),
    ("PAAD", "M4-SA", "M4", ("M2", "M4")),
    ("PAAD", "M4-SA-FIXC", "M4-FIXC", None),
    ("BRCA", "M3-SA", "M3", ("M1", "M3")),
    ("BRCA", "M4-SA", "M4", ("M2", "M4")),
]


def harrell(r, t, e):
    """Harrell's C, 동점 risk 0.5. 비교쌍: i가 사건이고 t_i < t_j."""
    comp = (t[:, None] < t[None, :]) & (e[:, None] == 1)
    d = r[:, None] - r[None, :]
    return (((d > 0) & comp).sum() + 0.5 * ((d == 0) & comp).sum()) / comp.sum()


def load(cohort, split, model):
    tag = afb.COHORTS[cohort]["models"][model]
    ids, r, t, e = afb.load(cohort, split, tag)
    return {c: (a, b, d) for c, a, b, d in zip(ids, r, t, e)}


def arrays(pred, ids):
    v = [pred[i] for i in ids if i in pred]
    return np.array([x[0] for x in v]), np.array([x[1] for x in v], float), np.array([x[2] for x in v], int)


def gap_effect(preds, int_ids, ext_ids):
    """preds: {model_key: {"internal": dict, "external": dict}}, A=self-attention, B=cross-attention."""
    out = {}
    for k in ("A", "B"):
        out[k + "_int"] = harrell(*arrays(preds[k]["internal"], int_ids))
        out[k + "_ext"] = harrell(*arrays(preds[k]["external"], ext_ids))
    out["d_int"] = out["B_int"] - out["A_int"]
    out["d_ext"] = out["B_ext"] - out["A_ext"]
    out["d_gap"] = out["d_ext"] - out["d_int"]
    # 2026-10-19 사용자 요청: 방향이 아니라 격차의 크기가 줄었는지. 음수면 |격차|가 줄어 internal과 external이 가까워짐
    out["d_absgap"] = abs(out["B_ext"] - out["B_int"]) - abs(out["A_ext"] - out["A_int"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.RandomState(0)
    rows = []
    for cohort, a, b, rna_pair in COMPARISONS:
        preds = {k: {s: load(cohort, s, m) for s in ("internal", "external")} for k, m in (("A", a), ("B", b))}
        int_ids = sorted(set(preds["A"]["internal"]) & set(preds["B"]["internal"]))
        ext_ids = sorted(set(preds["A"]["external"]) & set(preds["B"]["external"]))
        point = gap_effect(preds, int_ids, ext_ids)
        boots = []
        for _ in range(args.n_boot):
            if cohort == "BRCA":  # 같은 환자 집합을 internal·held-out에 함께
                pool = sorted(set(int_ids) | set(ext_ids))
                pick = [pool[i] for i in rng.randint(0, len(pool), len(pool))]
                bi = [c for c in pick if c in preds["A"]["internal"] and c in preds["B"]["internal"]]
                be = [c for c in pick if c in preds["A"]["external"] and c in preds["B"]["external"]]
            else:
                bi = [int_ids[i] for i in rng.randint(0, len(int_ids), len(int_ids))]
                be = [ext_ids[i] for i in rng.randint(0, len(ext_ids), len(ext_ids))]
            g = gap_effect(preds, bi, be)
            boots.append((g["d_gap"], g["d_absgap"]))
        boots = np.array(boots)
        lo, hi = np.percentile(boots[:, 0], [2.5, 97.5])
        p = min(1.0, 2 * min((boots[:, 0] <= 0).mean(), (boots[:, 0] >= 0).mean()))
        alo, ahi = np.percentile(boots[:, 1], [2.5, 97.5])
        ap_ = min(1.0, 2 * min((boots[:, 1] <= 0).mean(), (boots[:, 1] >= 0).mean()))
        rna = None
        if rna_pair:
            pr = {k: load(cohort, "external", m) for k, m in zip(("A", "B"), rna_pair)}
            ids = sorted(set(pr["A"]) & set(pr["B"]))
            rna = harrell(*arrays(pr["B"], ids)) - harrell(*arrays(pr["A"], ids))
        rows.append(dict(cohort=cohort, pair=f"{a} -> {b}", **point, lo=lo, hi=hi, p=p, alo=alo, ahi=ahi, ap=ap_, rna=rna,
                         rna_pair=" -> ".join(rna_pair) if rna_pair else "---"))
        print(f"{cohort} {a}->{b}: gap SA {point['A_ext'] - point['A_int']:+.4f} XA {point['B_ext'] - point['B_int']:+.4f} "
              f"d_gap {point['d_gap']:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.3f} | d_absgap {point['d_absgap']:+.4f} [{alo:+.4f}, {ahi:+.4f}] p={ap_:.3f}", flush=True)

    L = ["# RNA-guided cross-attention과 internal-external 격차", "",
         f"격차 = external C − internal C (BRCA는 held-out institution). Δgap = Δexternal − Δinternal, 양수면 격차가 줄어듦. "
         f"bootstrap {args.n_boot}회(PAAD internal·external 따로, BRCA 환자 단위 함께). Harrell's C, 5 seeds x 5 folds best+final 앙상블.", "",
         "| Cohort | Pair | SA int | SA ext | SA gap | XA int | XA ext | XA gap | Δint | Δext | Δgap [95% CI] | p | Δ|gap| [95% CI] | p | RNA-seq 추가 쌍 | 그 쌍의 Δext |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['cohort']} | {r['pair']} | {r['A_int']:.4f} | {r['A_ext']:.4f} | {r['A_ext'] - r['A_int']:+.4f} | "
                 f"{r['B_int']:.4f} | {r['B_ext']:.4f} | {r['B_ext'] - r['B_int']:+.4f} | {r['d_int']:+.4f} | {r['d_ext']:+.4f} | "
                 f"{r['d_gap']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['p']:.3f} | {r['d_absgap']:+.4f} [{r['alo']:+.4f}, {r['ahi']:+.4f}] | {r['ap']:.3f} | {r['rna_pair']} | "
                 + ("---" if r["rna"] is None else f"{r['rna']:+.4f}") + " |")
    out = _ROOT / "paper" / "results_coattn_gap.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
