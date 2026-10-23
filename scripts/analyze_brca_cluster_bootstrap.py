"""
2026-10-20: TCGA-BRCA 주요 통계 비교 전체의 Tissue Source Site(TSS) 단위 paired cluster bootstrap 민감도 분석.
원고·기존 결과 파일은 건드리지 않는다. 저장된 환자별 예측(5 seeds x 5 folds 앙상블)을 그대로 쓰고 재학습은 없다.

대상 (원고 번호. 분석 산출물의 옛 번호는 disp()로 변환)
  1. 모델별 C-index 95% CI (7개 + M4-SA, M7-SA, MMP), internal / held-out
  2. 쌍 비교 표(원고 BRCA Seven-Model Comparison 표): 11쌍 delta, p, split별 BH q(11개)
  3. gap 표: BRCA 2행 change in gap, p, q(PAAD 3행의 기존 p와 합친 5개 BH)
  4. 상호작용 표: M4 vs M7 cross-attention 효과 차이(internal, held-out, gap), q(PAAD 기존 p 3개와 합친 6개 BH)
  5. MMP 비교: MMP -> M4, MMP -> M7, internal / held-out

두 방식을 같은 엔진으로 나란히 계산한다.
  - patient-level: 기존 분석의 방식을 그대로 재현. 모델 CI·쌍 비교·MMP는 그 비교의 환자 집합 안에서 split별로 환자를
    복원추출, gap·상호작용은 두 split 환자 합집합을 한 번 뽑아 함께 사용(analyze_coattn_gap / analyze_xa_interaction과 동일).
  - cluster-level: 반복마다 39개 TSS를 복원추출 한 번. 뽑힌 TSS의 환자를 모두 넣고, 같은 TSS가 k번 뽑히면 그 환자들은
    가중치 k. 이 한 번의 기관 표본을 모든 모델·비교·두 split이 공유한다(paired).
가중치 Harrell C: sum_ij w_i w_j [concordant + 0.5 tie] * comparable_ij / sum_ij w_i w_j comparable_ij.
환자를 k번 복제해 보통 Harrell C를 계산한 것과 정확히 같다. 같은 환자의 복제본끼리는 OS_time이 같아(t_i < t_j 불성립)
비교쌍이 되지 않기 때문이다. comparable pair가 0이면 그 반복은 그 통계량에서 실패로 세고 제외한다.
p = 2 x min(P(Δ<=0), P(Δ>=0)), CI는 percentile.

사용법: python -m scripts.analyze_brca_cluster_bootstrap [--n-boot 2000]
출력: paper/results_brca_cluster_bootstrap.md, paper/results_brca_cluster_bootstrap_tss.tsv
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import analyze_baselines as ab  # noqa: E402
import analyze_foldsafe_batch as afb  # noqa: E402

SPLITS = ("internal", "external")
_NAME = {"1": "M1", "2": "M5", "3": "M4", "4": "M7", "5": "M3", "6": "M2", "7": "M6"}
GROUP = {"WSI": "WSI", "RNA-seq+CNV": "Molecular", "Clinical": "Clinicopathologic", "co-attention": "Cross-attention"}
SPLIT_LABEL = {"internal": "Internal", "external": "Held-out"}
# 원고에 실린 PAAD 쪽 p(BH family를 맞추기 위해). PAAD에는 TSS 단위 재추출을 하지 않는다(CPTAC에 기관 코드 없음).
PAAD_GAP_P = [0.952, 0.288, 0.462]
PAAD_INTERACTION_P = [0.713, 0.723, 0.624]


def disp(m):
    return re.sub(r"\bM([1-7])\b", lambda x: _NAME[x.group(1)], m)


def bh(p):
    return afb.bh(p)


def summarize(point, draws):
    d = np.asarray([x for x in draws if np.isfinite(x)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
    return dict(point=point, lo=lo, hi=hi, p=p, median=float(np.median(d)), n_ok=len(d), n_fail=len(draws) - len(d))


class Split:
    """한 split의 환자 universe와 모델별 pairwise 행렬. 가중치 C = (w K w) / (w Comp w)."""

    def __init__(self, preds):
        self.ids = sorted(set().union(*[set(p) for p in preds.values()]))
        self.pos = {c: i for i, c in enumerate(self.ids)}
        n = len(self.ids)
        t = np.full(n, np.nan)
        e = np.zeros(n, int)
        for p in preds.values():
            for c, (_, ti, ei) in p.items():
                i = self.pos[c]
                if np.isfinite(t[i]) and (abs(t[i] - ti) > 1e-6 or e[i] != ei):
                    raise SystemExit(f"라벨 불일치 {c}")
                t[i], e[i] = ti, ei
        self.t, self.e = t, e
        self.comp = ((t[:, None] < t[None, :]) & (e[:, None] == 1)).astype(np.float64)
        self.K, self.mask = {}, {}
        for m, p in preds.items():
            r = np.full(n, np.nan)
            for c, (ri, _, _) in p.items():
                r[self.pos[c]] = ri
            ok = np.isfinite(r)
            rr = np.where(ok, r, 0.0)
            K = self.comp * ((rr[:, None] > rr[None, :]) + 0.5 * (rr[:, None] == rr[None, :]))
            self.K[m] = K.astype(np.float64)
            self.mask[m] = ok.astype(np.float64)

    def common(self, models):
        m = np.ones(len(self.ids))
        for x in models:
            m = m * self.mask[x]
        return m

    def c(self, model, w):
        den = w @ self.comp @ w
        if den <= 0:
            return np.nan
        return float(w @ self.K[model] @ w / den)


def load_all():
    preds = {s: {} for s in SPLITS}
    for name, tag in afb.BRCA.items():
        for s in SPLITS:
            ids, r, t, e = afb.load("BRCA", s, tag)
            preds[s][name] = {c: (float(a), float(b), int(d)) for c, a, b, d in zip(ids, r, np.asarray(t, float), np.asarray(e, int))}
    mmp = [b for b in ab.BASELINES if b["name"] == "MMP"][0]
    for s in SPLITS:
        ids, r, t, e = ab.load_baseline(mmp, s, afb.SEEDS)
        preds[s]["MMP"] = {c: (float(a), float(b), int(d)) for c, a, b, d in zip(ids, r, t, e)}
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    B = args.n_boot
    preds = load_all()
    S = {s: Split(preds[s]) for s in SPLITS}
    models = list(afb.BRCA) + ["MMP"]
    pairs = [(GROUP[g], a, b) for g, a, b in afb.MODALITY_PAIRS]
    mmp_cmp = [("MMP", "M3"), ("MMP", "M4")]          # 원고 MMP -> M4, MMP -> M7
    gap_cmp = [("M3-SA", "M3"), ("M4-SA", "M4")]      # 원고 M4-SA -> M4, M7-SA -> M7
    inter = ("M3-SA", "M3", "M4-SA", "M4")            # 원고 M4 효과 − M7 효과

    # ---- 기관 구성과 불균형 ----
    pool = sorted(set(S["internal"].ids) | set(S["external"].ids))
    tss_of = {c: c.split("-")[1] for c in pool}
    tss = sorted(set(tss_of.values()))
    rows = []
    for k in tss:
        members = [c for c in pool if tss_of[c] == k]
        ext = [c for c in members if c in S["external"].pos]
        itn = [c for c in members if c in S["internal"].pos]
        rows.append(dict(tss=k, patients=len(members), events=int(sum(S["external"].e[S["external"].pos[c]] for c in ext)),
                         internal_patients=len(itn),
                         internal_events=int(sum(S["internal"].e[S["internal"].pos[c]] for c in itn))))
    tdf = pd.DataFrame(rows).sort_values("patients", ascending=False)
    tdf.to_csv(_ROOT / "paper" / "results_brca_cluster_bootstrap_tss.tsv", sep="\t", index=False)
    n_k = tdf["patients"].to_numpy()
    kish = n_k.sum() ** 2 / (n_k ** 2).sum()
    ev_k = tdf["events"].to_numpy()
    kish_ev = ev_k.sum() ** 2 / (ev_k ** 2).sum()
    members_idx = {s: {k: np.array([S[s].pos[c] for c in pool if tss_of[c] == k and c in S[s].pos], int) for k in tss} for s in SPLITS}

    # ---- 점추정 ----
    def point_c(s, m, mask):
        return S[s].c(m, mask)

    point = {}
    for s in SPLITS:
        for m in models:
            point[("model", s, m)] = point_c(s, m, S[s].mask[m])
        for _, a, b in pairs:
            w = S[s].common([a, b])
            point[("pair", s, a, b)] = point_c(s, b, w) - point_c(s, a, w)
        for a, b in mmp_cmp:
            w = S[s].common([a, b])
            point[("mmp", s, a, b)] = point_c(s, b, w) - point_c(s, a, w)

    def gap_stats(wi, we):
        """wi, we: internal/external 가중치(이미 환자 존재 여부 반영 전)."""
        out = {}
        for a, b in gap_cmp:
            mi, me = wi * S["internal"].common([a, b]), we * S["external"].common([a, b])
            di = S["internal"].c(b, mi) - S["internal"].c(a, mi)
            de = S["external"].c(b, me) - S["external"].c(a, me)
            out[("gap", a, b)] = de - di
        mi, me = wi * S["internal"].common(inter), we * S["external"].common(inter)
        x = {}
        for s, w in (("internal", mi), ("external", me)):
            x[s] = (S[s].c(inter[1], w) - S[s].c(inter[0], w)) - (S[s].c(inter[3], w) - S[s].c(inter[2], w))
        out[("int", "internal")] = x["internal"]
        out[("int", "external")] = x["external"]
        out[("int", "gap")] = x["external"] - x["internal"]
        return out

    ones = {s: np.ones(len(S[s].ids)) for s in SPLITS}
    point.update(gap_stats(ones["internal"], ones["external"]))

    # ---- bootstrap ----
    keys = list(point)
    draws = {"patient": {k: [] for k in keys}, "cluster": {k: [] for k in keys}}
    rng_p, rng_c = np.random.RandomState(0), np.random.RandomState(1)
    pool_pos = {s: np.array([S[s].pos.get(c, -1) for c in pool]) for s in SPLITS}

    def resample_within(s, mask):
        """mask가 1인 환자 안에서 환자 복원추출 → 가중치."""
        idx = np.flatnonzero(mask > 0)
        w = np.zeros(len(mask))
        np.add.at(w, idx[rng_p.randint(0, len(idx), len(idx))], 1.0)
        return w

    for it in range(B):
        # patient-level, 기존 방식 재현
        for s in SPLITS:
            for m in models:
                draws["patient"][("model", s, m)].append(S[s].c(m, resample_within(s, S[s].mask[m])))
            for _, a, b in pairs:
                w = resample_within(s, S[s].common([a, b]))
                draws["patient"][("pair", s, a, b)].append(S[s].c(b, w) - S[s].c(a, w))
            for a, b in mmp_cmp:
                w = resample_within(s, S[s].common([a, b]))
                draws["patient"][("mmp", s, a, b)].append(S[s].c(b, w) - S[s].c(a, w))
        pick = rng_p.randint(0, len(pool), len(pool))
        wj = {}
        for s in SPLITS:
            w = np.zeros(len(S[s].ids))
            pp = pool_pos[s][pick]
            np.add.at(w, pp[pp >= 0], 1.0)
            wj[s] = w
        for k, v in gap_stats(wj["internal"], wj["external"]).items():
            draws["patient"][k].append(v)

        # cluster-level, TSS 한 번 뽑아 모두 공유
        cnt = np.bincount(rng_c.randint(0, len(tss), len(tss)), minlength=len(tss)).astype(float)
        wc = {}
        for s in SPLITS:
            w = np.zeros(len(S[s].ids))
            for j, k in enumerate(tss):
                if cnt[j]:
                    w[members_idx[s][k]] = cnt[j]
            wc[s] = w
        for s in SPLITS:
            for m in models:
                draws["cluster"][("model", s, m)].append(S[s].c(m, wc[s] * S[s].mask[m]))
            for _, a, b in pairs:
                w = wc[s] * S[s].common([a, b])
                draws["cluster"][("pair", s, a, b)].append(S[s].c(b, w) - S[s].c(a, w))
            for a, b in mmp_cmp:
                w = wc[s] * S[s].common([a, b])
                draws["cluster"][("mmp", s, a, b)].append(S[s].c(b, w) - S[s].c(a, w))
        for k, v in gap_stats(wc["internal"], wc["external"]).items():
            draws["cluster"][k].append(v)
        if (it + 1) % 200 == 0:
            print(f"  {it + 1}/{B}", flush=True)

    res = {lvl: {k: summarize(point[k], draws[lvl][k]) for k in keys} for lvl in ("patient", "cluster")}

    # ---- BH q ----
    for lvl in ("patient", "cluster"):
        for s in SPLITS:
            ks = [("pair", s, a, b) for _, a, b in pairs]
            for k, q in zip(ks, bh([res[lvl][k]["p"] for k in ks])):
                res[lvl][k]["q"] = q
        ks = [("gap", a, b) for a, b in gap_cmp]
        q = bh(PAAD_GAP_P + [res[lvl][k]["p"] for k in ks])[len(PAAD_GAP_P):]
        for k, v in zip(ks, q):
            res[lvl][k]["q"] = v
        ks = [("int", "internal"), ("int", "external"), ("int", "gap")]
        q = bh(PAAD_INTERACTION_P + [res[lvl][k]["p"] for k in ks])[len(PAAD_INTERACTION_P):]
        for k, v in zip(ks, q):
            res[lvl][k]["q"] = v

    # ---- 기존 결과 파일과 대조(patient-level 재현 확인) ----
    pub_m = pd.read_csv(_ROOT / "paper" / "results_foldsafe_batch_models.tsv", sep="\t").query("cohort == 'BRCA'")
    pub_p = pd.read_csv(_ROOT / "paper" / "results_foldsafe_batch_pairs.tsv", sep="\t").query("cohort == 'BRCA'")
    chk = []
    for _, r in pub_m.iterrows():
        for s in SPLITS:
            k = ("model", s, r["model"])
            if k in res["patient"]:
                chk.append(("model", abs(r[f"{s}_c"] - res["patient"][k]["point"]),
                            max(abs(r[f"{s}_lo"] - res["patient"][k]["lo"]), abs(r[f"{s}_hi"] - res["patient"][k]["hi"]))))
    for _, r in pub_p.iterrows():
        a, b = [x.strip() for x in r["pair"].split("->")]
        k = ("pair", r["split"], a, b)
        if k in res["patient"]:
            chk.append(("pair", abs(r["delta"] - res["patient"][k]["point"]),
                        max(abs(r["lo"] - res["patient"][k]["lo"]), abs(r["hi"] - res["patient"][k]["hi"]))))
    chk = pd.DataFrame(chk, columns=["kind", "d_point", "d_ci"])

    # ---- 출력 ----
    def sig(x):
        return x["lo"] > 0 or x["hi"] < 0

    def cell(x, with_q=False):
        s = f"{x['point']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}] | {x['p']:.3f}"
        return s + (f" | {x['q']:.3f}" if with_q else "")

    def width(x):
        return x["hi"] - x["lo"]

    summary = []

    def flag(label, k, use_q):
        P, C = res["patient"][k], res["cluster"][k]
        sp = (P["q"] < 0.05) if use_q else sig(P)
        sc = (C["q"] < 0.05) if use_q else sig(C)
        flip = np.sign(C["median"]) != np.sign(C["point"]) and abs(C["point"]) > 1e-12
        summary.append(dict(label=label, ratio=width(C) / width(P), sp=sp, sc=sc, flip=flip,
                            fail=C["n_fail"], pfail=P["n_fail"], crit="q < 0.05" if use_q else "CI excludes 0"))
        return f"{width(C) / width(P):.2f}"

    L = ["# TCGA-BRCA 기관(TSS) 단위 paired cluster bootstrap 민감도 분석", "",
         f"bootstrap {B}회. patient-level은 기존 분석 방식 재현, cluster-level은 39개 TSS 복원추출(반복마다 한 번, 모든 모델·비교·split 공유). "
         "가중치 Harrell C(중복 추출된 TSS의 환자는 뽑힌 횟수만큼 가중). 원고 번호로 표기. 원고와 기존 결과 파일은 변경하지 않음. "
         "각 칸은 점추정 [95% CI] | p (| q). 점추정은 두 방식이 같다(같은 데이터).", ""]

    L += ["## 1. 기관 구성과 불균형", "",
          f"- TSS {len(tss)}곳, 환자 {int(n_k.sum())}명, 사건 {int(ev_k.sum())}건(held-out 예측 기준)",
          f"- 기관당 환자 수 중앙값 {np.median(n_k):.0f}, 범위 {n_k.min()}–{n_k.max()}, 상위 5곳이 전체의 {n_k[:5].sum() / n_k.sum():.0%}",
          f"- 사건이 0건인 기관 {int((ev_k == 0).sum())}곳, 사건 상위 5곳이 전체 사건의 {np.sort(ev_k)[::-1][:5].sum() / ev_k.sum():.0%}",
          f"- 유효 기관 수(Kish, 환자 기준) {kish:.1f}, 사건 기준 {kish_ev:.1f} — 명목 39곳보다 훨씬 적은 기관이 정보를 좌우",
          f"- internal 예측이 있는 환자 {int(tdf['internal_patients'].sum())}명, 사건 {int(tdf['internal_events'].sum())}건",
          "", "| TSS | 환자 | 사건 | internal 환자 | internal 사건 |", "|---|---|---|---|---|"]
    for _, r in tdf.iterrows():
        L.append(f"| {r['tss']} | {r['patients']} | {r['events']} | {r['internal_patients']} | {r['internal_events']} |")

    L += ["", "## 2. 모델별 C-index 95% CI", "",
          "| 모델 | Split | C | patient-level CI | cluster-level CI | 폭 비율 | 실패(c) |", "|---|---|---|---|---|---|---|"]
    order = ["M1", "M6", "M5", "M3", "M2", "M7", "M4", "M3-SA", "M4-SA", "MMP"]   # 원고 M1..M7, M4-SA, M7-SA, MMP
    for m in order:
        for s in SPLITS:
            k = ("model", s, m)
            P, C = res["patient"][k], res["cluster"][k]
            r_ = flag(f"모델 {disp(m)} {SPLIT_LABEL[s]}", k, False)
            L.append(f"| {disp(m)} | {SPLIT_LABEL[s]} | {P['point']:.4f} | [{P['lo']:.4f}, {P['hi']:.4f}] | [{C['lo']:.4f}, {C['hi']:.4f}] | {r_} | {C['n_fail']} |")

    L += ["", "## 3. 쌍 비교 (원고 BRCA Seven-Model Comparison 표), q는 split별 11개 BH", "",
          "| 추가 요소 | 쌍 | Split | patient-level Δ [CI] | p | q | cluster-level Δ [CI] | p | q | 폭 비율 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g, a, b in pairs:
        for s in SPLITS:
            k = ("pair", s, a, b)
            r_ = flag(f"{g} {disp(a)}→{disp(b)} {SPLIT_LABEL[s]}", k, True)
            L.append(f"| {g} | {disp(a)} → {disp(b)} | {SPLIT_LABEL[s]} | {cell(res['patient'][k], True)} | {cell(res['cluster'][k], True)} | {r_} |")

    L += ["", "## 4. Internal-to-external gap (원고 gap 표 BRCA 2행), q는 PAAD 기존 p 3개와 합친 5개 BH", "",
          "| 비교 | patient-level Δgap [CI] | p | q | cluster-level Δgap [CI] | p | q | 폭 비율 |", "|---|---|---|---|---|---|---|---|"]
    for a, b in gap_cmp:
        k = ("gap", a, b)
        r_ = flag(f"gap {disp(a)}→{disp(b)}", k, True)
        L.append(f"| {disp(a)} → {disp(b)} | {cell(res['patient'][k], True)} | {cell(res['cluster'][k], True)} | {r_} |")

    L += ["", "## 5. Cross-attention × clinicopathologic 상호작용 (원고 상호작용 표 BRCA 3행), q는 PAAD 기존 p 3개와 합친 6개 BH", "",
          "| 지표 | patient-level [CI] | p | q | cluster-level [CI] | p | q | 폭 비율 |", "|---|---|---|---|---|---|---|---|"]
    for lab, k in (("Internal C", ("int", "internal")), ("Held-out C", ("int", "external")), ("Change in gap", ("int", "gap"))):
        r_ = flag(f"상호작용 {lab}", k, True)
        L.append(f"| {lab} | {cell(res['patient'][k], True)} | {cell(res['cluster'][k], True)} | {r_} |")

    L += ["", "## 6. MMP baseline 비교 (원고 baseline 표 BRCA 4행, 보정 없음)", "",
          "| 비교 | Split | patient-level Δ [CI] | p | cluster-level Δ [CI] | p | 폭 비율 |", "|---|---|---|---|---|---|---|"]
    for a, b in mmp_cmp:
        for s in SPLITS:
            k = ("mmp", s, a, b)
            r_ = flag(f"MMP→{disp(b)} {SPLIT_LABEL[s]}", k, False)
            L.append(f"| MMP → {disp(b)} | {SPLIT_LABEL[s]} | {cell(res['patient'][k])} | {cell(res['cluster'][k])} | {r_} |")

    sm = pd.DataFrame(summary)
    lost = sm[sm.sp & ~sm.sc]
    gained = sm[~sm.sp & sm.sc]
    L += ["", "## 7. 요약", "",
          f"- 비교한 통계량 {len(sm)}개. CI 폭 비율(cluster/patient) 중앙값 {sm.ratio.median():.2f}, 범위 {sm.ratio.min():.2f}–{sm.ratio.max():.2f}",
          f"- 유의성 기준: 쌍·gap·상호작용은 q < 0.05, 모델 CI·MMP는 CI가 0을 제외(모델 CI는 해당 없음)",
          f"- patient-level에서 유의했다가 cluster-level에서 유의하지 않게 된 것 {len(lost)}개" + ("" if lost.empty else ": " + "; ".join(lost.label)),
          f"- 반대로 새로 유의해진 것 {len(gained)}개" + ("" if gained.empty else ": " + "; ".join(gained.label)),
          f"- 방향 반전(cluster bootstrap 중앙값의 부호가 점추정과 반대) {int(sm.flip.sum())}개" + ("" if not sm.flip.any() else ": " + "; ".join(sm[sm.flip].label)),
          f"- 실패한 반복(comparable pair 0): cluster-level 최대 {int(sm.fail.max())}회, patient-level 최대 {int(sm.pfail.max())}회 / {B}회",
          "", "### patient-level 재현 확인 (기존 results_foldsafe_batch_*.tsv 대비)", "",
          f"- 점추정 최대 차이 {chk.d_point.max():.2e}(같은 데이터라 0이어야 함), CI 경계 최대 차이 {chk.d_ci.max():.4f}(난수 차이), 대조 {len(chk)}개"]
    out = _ROOT / "paper" / "results_brca_cluster_bootstrap.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-12:]))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
