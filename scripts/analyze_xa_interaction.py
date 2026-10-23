"""
2026-10-20: GPT 사전 검토 P1 후속 — RNA-guided cross-attention 효과가 clinicopathologic 변수 유무에 따라
다른지 하나의 검정으로 직접 본다(attention × clinicopathologic 상호작용, difference-in-differences).

    Δ = [C(M4) − C(M4-SA)] − [C(M7) − C(M7-SA)]      (원고 번호. M4는 임상정보 없음, M7은 있음)

held-out(external) C, internal C, 격차 변화(Δext − Δint) 각각에 대해 계산한다.
BRCA: 네 모델이 같은 환자를 평가하므로 환자를 한 번 뽑아 네 모델·두 split에 함께 쓴다.
PAAD: internal(TCGA)과 external(CPTAC)은 다른 환자라 따로 뽑되, 각 split 안에서는 네 모델에 같은 환자를 쓴다.
분석 산출물의 옛 번호: 원고 M4 = 옛 M3, 원고 M7 = 옛 M4.

2026-10-20 추가(GPT 사전 검토): --cluster-tss는 BRCA를 환자 대신 Tissue Source Site(바코드 둘째 칸, 39곳) 단위로
복원추출하는 민감도 분석. 뽑힌 기관의 환자를 모두 함께 넣는다(같은 기관이 여러 번 뽑히면 환자도 그만큼 중복).
기관 내 상관을 반영하므로 CI가 넓어질 수 있다. 여섯 상호작용 검정의 BH q와 Bonferroni 보정 p도 함께 적는다.

사용법: python -m scripts.analyze_xa_interaction [--n-boot 2000]                -> paper/results_xa_interaction.md
        python -m scripts.analyze_xa_interaction --cluster-tss [--n-boot 2000]  -> paper/results_xa_interaction_cluster.md (BRCA만)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from analyze_coattn_gap import arrays, harrell, load  # noqa: E402

MODELS = {"SA4": "M3-SA", "XA4": "M3", "SA7": "M4-SA", "XA7": "M4"}  # 분석 ID


def effects(p, int_ids, ext_ids):
    c = {(k, s): harrell(*arrays(p[k][s], ids)) for k in MODELS for s, ids in (("internal", int_ids), ("external", ext_ids))}
    out = {}
    for s in ("internal", "external"):
        out[f"xa4_{s}"] = c[("XA4", s)] - c[("SA4", s)]
        out[f"xa7_{s}"] = c[("XA7", s)] - c[("SA7", s)]
        out[f"int_{s}"] = out[f"xa4_{s}"] - out[f"xa7_{s}"]
    out["gap4"] = out["xa4_external"] - out["xa4_internal"]
    out["gap7"] = out["xa7_external"] - out["xa7_internal"]
    out["int_gap"] = out["gap4"] - out["gap7"]
    return out


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
    return lo, hi, p


def bh(ps):
    """Benjamini-Hochberg q-value."""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    q = ps[order] * len(ps) / np.arange(1, len(ps) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--cluster-tss", action="store_true", help="BRCA만, Tissue Source Site 단위 cluster bootstrap")
    args = ap.parse_args()
    rng = np.random.RandomState(0)
    keys = ["xa4_external", "xa7_external", "int_external", "xa4_internal", "xa7_internal", "int_internal", "gap4", "gap7", "int_gap"]
    rows = []
    for cohort in (("BRCA",) if args.cluster_tss else ("BRCA", "PAAD")):
        p = {k: {s: load(cohort, s, m) for s in ("internal", "external")} for k, m in MODELS.items()}
        int_ids = sorted(set.intersection(*[set(p[k]["internal"]) for k in MODELS]))
        ext_ids = sorted(set.intersection(*[set(p[k]["external"]) for k in MODELS]))
        point = effects(p, int_ids, ext_ids)
        boots = {k: [] for k in keys}
        pool = sorted(set(int_ids) | set(ext_ids))
        by_tss = {}
        for c in pool:
            by_tss.setdefault(c.split("-")[1], []).append(c)
        tss = sorted(by_tss)
        for _ in range(args.n_boot):
            if args.cluster_tss:
                pick = [c for i in rng.randint(0, len(tss), len(tss)) for c in by_tss[tss[i]]]
                si, se = set(int_ids), set(ext_ids)
                bi = [c for c in pick if c in si]
                be = [c for c in pick if c in se]
            elif cohort == "BRCA":
                pool = sorted(set(int_ids) | set(ext_ids))
                pick = [pool[i] for i in rng.randint(0, len(pool), len(pool))]
                si, se = set(int_ids), set(ext_ids)
                bi = [c for c in pick if c in si]
                be = [c for c in pick if c in se]
            else:
                bi = [int_ids[i] for i in rng.randint(0, len(int_ids), len(int_ids))]
                be = [ext_ids[i] for i in rng.randint(0, len(ext_ids), len(ext_ids))]
            e = effects(p, bi, be)
            for k in keys:
                boots[k].append(e[k])
        res = {k: (point[k], *ci(np.array(boots[k]))) for k in keys}
        rows.append((cohort, len(int_ids), len(ext_ids), res))
        for k in keys:
            v = res[k]
            print(f"{cohort} {k:13s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] p={v[3]:.3f}", flush=True)

    # 상호작용 검정(코호트 x internal·external·격차)의 다중비교 보정
    ikeys = [(c, k) for c, _, _, res in rows for k in ("int_internal", "int_external", "int_gap")]
    ip = [res[k][3] for _, _, _, res in rows for k in ("int_internal", "int_external", "int_gap")]
    iq = dict(zip(ikeys, bh(ip)))
    ibon = dict(zip(ikeys, np.minimum(np.array(ip) * len(ip), 1.0)))
    for (c, k) in ikeys:
        print(f"{c} {k}: q={iq[(c, k)]:.3f} bonferroni={ibon[(c, k)]:.3f}")

    label = {"external": "Held-out / external C", "internal": "Internal C"}
    f = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] | {v[3]:.3f}"
    unit = ("Tissue Source Site 단위 cluster bootstrap(BRCA 39곳, 뽑힌 기관의 환자 전부, 두 split 함께)" if args.cluster_tss
            else "paired bootstrap(BRCA는 환자 단위로 두 split 함께, PAAD는 split별로 따로, 각 resample 안에서 네 모델에 같은 환자)")
    L = ["# Cross-attention × clinicopathologic 상호작용" + (" — TSS cluster bootstrap 민감도 분석" if args.cluster_tss else ""), "",
         "Δ = [C(M4) − C(M4-SA)] − [C(M7) − C(M7-SA)]. 원고 번호. M4는 clinicopathologic 없음, M7은 있음. "
         f"Harrell's C, {unit} {args.n_boot}회. "
         f"격차 변화 = Δexternal − Δinternal, 양수면 cross-attention이 격차를 줄임. 상호작용 q는 이 표의 상호작용 {len(ip)}개 BH, Bonferroni는 p×{len(ip)}.", "",
         "| Cohort | 지표 | M4 cross-attention 효과 [95% CI] | p | M7 cross-attention 효과 [95% CI] | p | 상호작용 M4 − M7 [95% CI] | p | q | Bonferroni |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for cohort, ni, ne, res in rows:
        for s, k in (("external", "int_external"), ("internal", "int_internal")):
            L.append(f"| {cohort} | {label[s]} | {f(res[f'xa4_{s}'])} | {f(res[f'xa7_{s}'])} | {f(res[k])} | {iq[(cohort, k)]:.3f} | {ibon[(cohort, k)]:.3f} |")
        L.append(f"| {cohort} | 격차 변화 | {f(res['gap4'])} | {f(res['gap7'])} | {f(res['int_gap'])} | {iq[(cohort, 'int_gap')]:.3f} | {ibon[(cohort, 'int_gap')]:.3f} |")
    L += ["", "N: " + ", ".join(f"{c} internal {ni} / external {ne}" for c, ni, ne, _ in rows)]
    out = _ROOT / "paper" / ("results_xa_interaction_cluster.md" if args.cluster_tss else "results_xa_interaction.md")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
