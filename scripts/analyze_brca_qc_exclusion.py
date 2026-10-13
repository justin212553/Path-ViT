"""
2026-10-12(D3 사전 점검, 리뷰 #13 데이터 QC): BRCA OS_time <= 0 환자 21명(음수 1명, 0일 20명 중 사망 2명)을
기존 기관 단위 CV(INSTCV_FS) 예측에서 평가 단계에서만 빼면 결과가 얼마나 바뀌는지 본다.

재학습이 아니다. 학습에 들어간 21명의 영향은 그대로 남아 있고, 평가 집합에서만 빠진다.
재실행(--external-tss instcvqc, 배정은 그대로 두고 21명만 제외) 전에 영향 크기를 가늠하려는 용도.

방법: .logs/{kfold,external}_preds의 BRCA 예측 파일에서 21명 행만 지운 사본을 임시 폴더에 만들고,
analyze_foldsafe_batch와 같은 집계(5 seed x 5 fold, best+final, union seeds, paired bootstrap 2000, BH)를
원본과 사본에 똑같이 돌려 나란히 비교한다. MMP 베이스라인(M4 vs MMP)도 함께 본다.

사용법: python -m scripts.analyze_brca_qc_exclusion   -> paper/results_brca_qc_exclusion.md
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import analyze_foldsafe_batch as afb  # noqa: E402
from paired_bootstrap_delta import _ensemble_external, _ensemble_internal, paired_bootstrap_delta  # noqa: E402
from utils.metrics import compute_survival_metrics  # noqa: E402

MODELS = dict(afb.BRCA, MMP="MMP_INSTCV")
PAIRS = afb.MODALITY_PAIRS + [("baseline", "MMP", "M4")]


def excluded_ids():
    clin = pd.read_csv(_ROOT / "data" / "brca_clinical.csv")
    return set(clin.loc[clin["OS_time"] <= 0, "case_id"])


def make_filtered_copy(dst: Path, drop: set):
    n_rows = 0
    for sub in ("kfold_preds", "external_preds"):
        (dst / sub).mkdir(parents=True)
        for f in (afb.PRED / sub).glob("brca_*INSTCV*.csv"):
            df = pd.read_csv(f)
            keep = ~df["case_id"].isin(drop)
            n_rows += int((~keep).sum())
            df[keep].to_csv(dst / sub / f.name, index=False)
    return n_rows


def run(pred_root: Path):
    afb.PRED = pred_root
    cache, models, pairs = {}, [], []
    for name, tag in MODELS.items():
        row = {"model": name}
        for split in ("internal", "external"):
            if name == "MMP":  # MMP는 FINALEPOCH 파일이 없다(analyze_baselines와 같은 처리)
                if split == "internal":
                    ids, r, t, e = _ensemble_internal(pred_root / "kfold_preds", "brca", tag, afb.SEEDS, afb.N_FOLDS,
                                                      include_final_epoch=False, union_seeds=True)
                else:
                    ids, r, t, e = _ensemble_external(pred_root / "external_preds", "brca", tag, afb.SEEDS,
                                                      afb.N_FOLDS, include_final_epoch=False)
            else:
                ids, r, t, e = afb.load("BRCA", split, tag)
            cache[(split, name)] = (ids, r, t, e)
            row[f"{split}_n"], row[f"{split}_events"] = len(ids), int(e.sum())
            row[f"{split}_c"] = compute_survival_metrics(r, t, e)["c_index"]
        models.append(row)
    for group, a, b in PAIRS:
        for split in ("internal", "external"):
            res = paired_bootstrap_delta(*cache[(split, a)], *cache[(split, b)], n_bootstrap=afb.N_BOOT)
            pairs.append({"group": group, "pair": f"{a} -> {b}", "split": split,
                          "delta": res["point_delta"], "p": res["p_value"]})
    pairs = pd.DataFrame(pairs)
    pairs["q"] = np.nan
    for split, idx in pairs.groupby("split").groups.items():
        pairs.loc[idx, "q"] = afb.bh(pairs.loc[idx, "p"])
    return pd.DataFrame(models), pairs


def main():
    drop = excluded_ids()
    orig_m, orig_p = run(_ROOT / ".logs")
    with tempfile.TemporaryDirectory() as td:
        n_rows = make_filtered_copy(Path(td), drop)
        qc_m, qc_p = run(Path(td))

    m = orig_m.merge(qc_m, on="model", suffixes=("", "_qc"))
    p = orig_p.merge(qc_p, on=["group", "pair", "split"], suffixes=("", "_qc"))
    lines = ["# BRCA OS_time <= 0 제외 영향 (평가 단계만, 재학습 없음)", "",
             f"- 제외 대상 {len(drop)}명, 예측 파일에서 지운 행 {n_rows}개",
             "- 집계는 analyze_foldsafe_batch와 동일(5 seed x 5 fold, best+final, union seeds, bootstrap 2000, BH는 split별)",
             "- 학습에 들어간 21명의 영향은 남아 있으므로 재실행 결과의 근사치다", "",
             "## 모델별 Harrell's C", "",
             "| Model | Int N/events | Int C | Int C (QC) | Δ | Ext N/events | Ext C | Ext C (QC) | Δ |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in m.iterrows():
        lines.append(
            f"| {r.model} | {r.internal_n}/{r.internal_events} → {r.internal_n_qc}/{r.internal_events_qc} "
            f"| {r.internal_c:.4f} | {r.internal_c_qc:.4f} | {r.internal_c_qc - r.internal_c:+.4f} "
            f"| {r.external_n}/{r.external_events} → {r.external_n_qc}/{r.external_events_qc} "
            f"| {r.external_c:.4f} | {r.external_c_qc:.4f} | {r.external_c_qc - r.external_c:+.4f} |")
    lines += ["", "## 쌍 비교 (paired bootstrap)", "",
              "| Group | Pair | Split | ΔC | p | q | ΔC (QC) | p (QC) | q (QC) |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in p.iterrows():
        lines.append(f"| {r.group} | {r.pair} | {r.split} | {r.delta:+.4f} | {r.p:.3f} | {r.q:.3f} "
                     f"| {r.delta_qc:+.4f} | {r.p_qc:.3f} | {r.q_qc:.3f} |")
    dm = pd.concat([m.internal_c_qc - m.internal_c, m.external_c_qc - m.external_c]).abs()
    dp = (p.delta_qc - p.delta).abs()
    flips = p[(p.q < 0.05) != (p.q_qc < 0.05)]
    lines += ["", "## 요약", "",
              f"- 모델 C 변화 |Δ|: 중간값 {dm.median():.4f}, 최대 {dm.max():.4f}",
              f"- 쌍 ΔC 변화 |Δ|: 중간값 {dp.median():.4f}, 최대 {dp.max():.4f}",
              f"- q < 0.05 판정이 바뀐 쌍: {len(flips)}개"
              + ("" if flips.empty else " — " + ", ".join(f"{r.pair} {r.split}" for _, r in flips.iterrows()))]
    out = _ROOT / "paper" / "results_brca_qc_exclusion.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-4:]))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
