"""
2026-10-04: 리뷰 대응([정찬권 #9]) — fold-safe 배치 예측에 Harrell's C 외 Uno's C(IPCW, 절단 tau)와
cumulative/dynamic time-dependent AUC를 추가 계산. 재학습 불필요(저장된 risk만 사용).

- 앙상블/코호트/모델 정의는 analyze_foldsafe_batch.py와 동일(load 재사용)
- 검열 분포 G(t)는 평가 집합 자체의 Kaplan-Meier(검열을 사건으로) — 코호트마다 추적 방식이 달라서.
  bootstrap에서는 resample마다 G를 다시 적합
- tau: PAAD 3년, BRCA 5년 (G(tau) >= 0.28로 가중치 폭주 방지). td-AUC 시점: PAAD 1/2/3년, BRCA 1/3/5년
- Uno's C와 Harrell's C 비교쌍 delta: 환자 단위 paired bootstrap, p는 paired_bootstrap_delta와 같은 정의

출력: paper/results_uno_tdauc.md, paper/results_uno_tdauc_models.tsv, _pairs.tsv
사용법: python scripts/analyze_uno_tdauc.py [--n-boot 1000] [--report-only]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from analyze_foldsafe_batch import COHORTS, load, bh  # noqa: E402

TAU = {"PAAD": 1095.0, "BRCA": 1825.0}
AUC_TIMES = {"PAAD": [365.0, 730.0, 1095.0], "BRCA": [365.0, 1095.0, 1825.0]}


def censor_km_left(t, e, q):
    """검열 생존함수 G(q-) — 검열(e==0)을 사건으로 본 KM의 좌극한."""
    ut = np.unique(t[e == 0])
    at_risk = np.array([(t >= u).sum() for u in ut])
    n_cens = np.array([((t == u) & (e == 0)).sum() for u in ut])
    surv = np.cumprod(1.0 - n_cens / at_risk)
    idx = np.searchsorted(ut, q, side="left") - 1  # q보다 엄격히 작은 마지막 검열 시점
    return np.where(idx >= 0, surv[np.clip(idx, 0, None)], 1.0)


def harrell_c(r, t, e):
    comp = (t[:, None] < t[None, :]) & (e[:, None] == 1)
    d = r[:, None] - r[None, :]
    return ((d > 0) & comp).sum() + 0.5 * ((d == 0) & comp).sum(), comp.sum()


def uno_c(r, t, e, tau):
    g = censor_km_left(t, e, t)
    w = np.where((e == 1) & (t < tau) & (g > 0), 1.0 / np.maximum(g, 1e-12) ** 2, 0.0)
    comp = (t[:, None] < t[None, :]) * w[:, None]
    d = r[:, None] - r[None, :]
    num = (comp * ((d > 0) + 0.5 * (d == 0))).sum()
    return num / comp.sum()


def td_auc(r, t, e, s):
    g = censor_km_left(t, e, t)
    case_w = np.where((t <= s) & (e == 1) & (g > 0), 1.0 / np.maximum(g, 1e-12), 0.0)
    ctrl = t > s
    if case_w.sum() == 0 or ctrl.sum() == 0:
        return np.nan
    d = r[:, None] - r[ctrl][None, :]
    num = (case_w[:, None] * ((d > 0) + 0.5 * (d == 0))).sum()
    return num / (case_w.sum() * ctrl.sum())


def metrics(r, t, e, cohort):
    n, d = harrell_c(r, t, e)
    out = {"harrell": n / d, "uno": uno_c(r, t, e, TAU[cohort])}
    for s in AUC_TIMES[cohort]:
        out[f"auc_{int(s / 365)}y"] = td_auc(r, t, e, s)
    return out


def align(A, B):
    da = dict(zip(A[0], zip(A[1], A[2], A[3])))
    db = dict(zip(B[0], B[1]))
    ids = sorted(set(da) & set(db))
    ra = np.array([da[i][0] for i in ids]); rb = np.array([db[i] for i in ids])
    t = np.array([da[i][1] for i in ids], float); e = np.array([da[i][2] for i in ids], int)
    return ra, rb, t, e


def _range_line(label, d):
    return f"- {label}: 중간값 {np.median(d):+.4f}, 범위 [{d.min():+.4f}, {d.max():+.4f}] (n={len(d)})"


def write_report(models, pairs, n_boot):
    out = _ROOT / "paper"
    lines = ["# Uno's C / time-dependent AUC (fold-safe 배치, 2026-10-04)", "",
             f"앙상블은 `results_foldsafe_batch.md`와 동일. Uno's C 절단 tau: PAAD {TAU['PAAD']:.0f}일(3년), "
             f"BRCA {TAU['BRCA']:.0f}일(5년). 검열 분포는 평가 집합 자체 KM. 비교쌍 bootstrap {n_boot}회, "
             "q = BH-FDR(코호트 x split별). Harrell's C 원값은 `results_foldsafe_batch.md` 참조.", ""]
    for cohort in COHORTS:
        m = models[models.cohort == cohort]
        d_int = (m["internal_uno"] - m["internal_harrell"]).to_numpy()
        d_ext = (m["external_uno"] - m["external_harrell"]).to_numpy()
        lines += [f"## {cohort} — Uno's C − Harrell's C (모델별)", "",
                  "| model | internal Δ | external Δ |", "|---|---|---|"]
        for (_, r), di, de in zip(m.iterrows(), d_int, d_ext):
            lines.append(f"| {r.model} | {di:+.4f} | {de:+.4f} |")
        lines += ["", f"요약 ({cohort}, 모델 {len(m)}개):",
                  _range_line("internal", d_int), _range_line("external", d_ext),
                  _range_line("internal+external", np.concatenate([d_int, d_ext])), ""]

        yrs = [f"auc_{int(s / 365)}y" for s in AUC_TIMES[cohort]]
        lines += [f"## {cohort} — Uno's C와 time-dependent AUC (모델별)", "",
                  "| model | Uno C int | Uno C ext | " + " | ".join(f"AUC {y[4:]} int | AUC {y[4:]} ext" for y in yrs) + " |",
                  "|---|---|---|" + "---|---|" * len(yrs)]
        for _, r in m.iterrows():
            lines.append(f"| {r.model} | {r.internal_uno:.4f} | {r.external_uno:.4f} | "
                         + " | ".join(f"{r[f'internal_{y}']:.4f} | {r[f'external_{y}']:.4f}" for y in yrs) + " |")
        lines.append("")

        lines += [f"## {cohort} — 비교쌍 ΔC (A -> B), Harrell vs Uno", "",
                  "| 그룹 | 비교 | split | Harrell Δ [95% CI] p (q) | Uno Δ [95% CI] p (q) |", "|---|---|---|---|---|"]
        for _, x in pairs[pairs.cohort == cohort].iterrows():
            cells = []
            for kind in ("harrell", "uno"):
                star = "**" if (x[f"{kind}_lo"] > 0 or x[f"{kind}_hi"] < 0) else ""
                cells.append(f"{star}{x[f'{kind}_delta']:+.4f} [{x[f'{kind}_lo']:+.4f}, {x[f'{kind}_hi']:+.4f}]{star} "
                             f"{x[f'{kind}_p']:.3f} ({x[f'{kind}_q']:.3f})")
            lines.append(f"| {x.group} | {x.pair} | {x.split} | {cells[0]} | {cells[1]} |")
        lines.append("")
    (out / "results_uno_tdauc.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--report-only", action="store_true",
                    help="재계산 없이 기존 _models.tsv/_pairs.tsv로 md 보고서만 다시 씀")
    args = ap.parse_args()
    if args.report_only:
        out = _ROOT / "paper"
        write_report(pd.read_csv(out / "results_uno_tdauc_models.tsv", sep="\t"),
                     pd.read_csv(out / "results_uno_tdauc_pairs.tsv", sep="\t"), args.n_boot)
        print("written paper/results_uno_tdauc.md")
        return

    cache, model_rows, pair_rows = {}, [], []
    for cohort, c in COHORTS.items():
        for name, tag in c["models"].items():
            row = {"cohort": cohort, "model": name}
            for split in ("internal", "external"):
                ids, r, t, e = load(cohort, split, tag)
                cache[(cohort, split, name)] = (ids, r, t, np.asarray(e, int))
                for k, v in metrics(r, np.asarray(t, float), np.asarray(e, int), cohort).items():
                    row[f"{split}_{k}"] = v
            model_rows.append(row)
            print(f"{cohort} {name:11s} ext harrell {row['external_harrell']:.4f} uno {row['external_uno']:.4f}", flush=True)

        rng = np.random.RandomState(0)
        for group, a, b in c["pairs"]:
            for split in ("internal", "external"):
                ra, rb, t, e = align(cache[(cohort, split, a)], cache[(cohort, split, b)])
                res = {"cohort": cohort, "group": group, "pair": f"{a} -> {b}", "split": split, "n": len(t)}
                for kind in ("harrell", "uno"):
                    f = (lambda r, t, e: harrell_c(r, t, e)[0] / harrell_c(r, t, e)[1]) if kind == "harrell" \
                        else (lambda r, t, e: uno_c(r, t, e, TAU[cohort]))
                    res[f"{kind}_delta"] = f(rb, t, e) - f(ra, t, e)
                deltas = {"harrell": [], "uno": []}
                for _ in range(args.n_boot):
                    ix = rng.randint(0, len(t), len(t))
                    if e[ix].sum() == 0:
                        continue
                    hb, hd = harrell_c(rb[ix], t[ix], e[ix]); ha, _ = harrell_c(ra[ix], t[ix], e[ix])
                    deltas["harrell"].append((hb - ha) / hd)
                    deltas["uno"].append(uno_c(rb[ix], t[ix], e[ix], TAU[cohort]) - uno_c(ra[ix], t[ix], e[ix], TAU[cohort]))
                for kind, ds in deltas.items():
                    ds = np.asarray(ds)
                    res[f"{kind}_lo"], res[f"{kind}_hi"] = np.percentile(ds, [2.5, 97.5])
                    res[f"{kind}_p"] = min(2 * min(np.mean(ds <= 0), np.mean(ds >= 0)), 1.0)
                pair_rows.append(res)
            print(f"  {cohort} {group} {a}->{b} done", flush=True)

    models = pd.DataFrame(model_rows)
    pairs = pd.DataFrame(pair_rows)
    for kind in ("harrell", "uno"):
        pairs[f"{kind}_q"] = np.nan
        for _, idx in pairs.groupby(["cohort", "split"]).groups.items():
            pairs.loc[idx, f"{kind}_q"] = bh(pairs.loc[idx, f"{kind}_p"])

    out = _ROOT / "paper"
    models.to_csv(out / "results_uno_tdauc_models.tsv", sep="\t", index=False)
    pairs.to_csv(out / "results_uno_tdauc_pairs.tsv", sep="\t", index=False)

    write_report(models, pairs, args.n_boot)
    print("written paper/results_uno_tdauc.md")


if __name__ == "__main__":
    main()
