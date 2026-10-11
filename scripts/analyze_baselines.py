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

# 2026-10-11: 두 수준으로 보고. 시스템 수준 = 각 baseline이 원작 입력 그대로(프로토콜만 우리와 동일),
# 아키텍처 수준 = 입력 유전자 세트까지 통일. PAAD는 PORPOISE에 우리 패널(C안), BRCA는 MMP가 RNA를 pathway token으로
# 받아 패널 교체가 불가하므로 반대로 우리 모델에 MMP의 Hallmark 유전자 세트(M3-HALL, M4-HALL).
def _resolve_hall(mid):
    """HALL 뒤 유전자 수는 HPC 매핑 결과로 정해지므로 파일명에서 태그를 찾는다(없으면 None)."""
    import re
    pat = re.compile(r"^brca_(BRCA_PMA_HALL\d+_" + re.escape(mid) + r"_NLLSURV4_NLLCOX1_INSTCV_FS)_seed\d+_fold\dof5\.csv$")
    tags = {m.group(1) for f in (PRED / "kfold_preds").glob("brca_BRCA_PMA_HALL*") if (m := pat.match(f.name))}
    return sorted(tags)[0] if len(tags) == 1 else None


OURS = {"PAAD": dict(PAAD), "BRCA": dict(BRCA)}
for _name, _mid in (("M3-HALL", "NOCLINICAL_SS_CLUSTERPOOL"), ("M4-HALL", "STG_SS_CLUSTERPOOL_CLR100")):
    _t = _resolve_hall(_mid)
    if _t:
        OURS["BRCA"][_name] = _t

BASELINES = [
    dict(cohort="PAAD", level="시스템 수준 — 원작 입력(공식 RNA 1553+CNV 82+mutation 14), 우리 코호트 중 95명(공식 입력 103명 중 ResNet50 feature 보유). "
               "CPTAC CNV·mutation은 cBioPortal pancreas_cptac_gdc 실측(CNV 79/82, mutation 14/14)",
         name="PORPOISE", tag="PORPOISE_OFFICIAL_N110_MMF", ext_tag="PORPOISE_OFFICIAL_N110_CBIO_MMF",
         internal="tcga", external="cptac", union=False, compare=["M3-noCNV", "M3", "M4", "M4-SA", "M7"]),
    dict(cohort="PAAD", level="(참고) 시스템 수준, CPTAC CNV·mutation 근사판(CNV 82개 neutral, mutation 10/14개 0) — 실측판으로 대체됨",
         name="PORPOISE", tag="PORPOISE_OFFICIAL_N110_MMF", internal="tcga", external="cptac", union=False,
         compare=["M4"], external_only=True),
    dict(cohort="PAAD", level="아키텍처 수준 — 입력 통일(PORPOISE에 우리 RNA 패널·코호트 110명, C안)",
         name="PORPOISE", tag="PORPOISE_N110_MMF", internal="tcga", external="cptac", union=False,
         compare=["M3-noCNV", "M3", "M4", "M4-SA", "M7"]),
    dict(cohort="BRCA", level="시스템 수준 — 원작 입력(MMP 공식 Hallmark RNA, 우리 모델은 CONS882)",
         name="MMP", tag="MMP_INSTCV", internal="brca", external="brca", union=True,
         compare=["M3", "M4", "M4-SA", "M7"]),
    # BRCA 아키텍처 수준(우리 모델 + Hallmark 유전자, M3HALL/M4HALL)은 2026-10-11 사용자 결정으로 보류(리뷰 지적 아님)
]
# 같은 아키텍처에서 패널만 교체(우리 패널의 기여 검증) — A -> B, Δ = B − A
PANEL_PAIRS = []  # 패널 기여 검증은 보류(리뷰 지적 아님, 2026-10-11). 재개 시 [("BRCA", "M3-HALL", "M3"), ("BRCA", "M4-HALL", "M4")]


def load_baseline(b, split, seeds):
    if split == "internal":
        ids, r, t, e = _ensemble_internal(PRED / "kfold_preds", b["internal"], b["tag"], seeds, 5,
                                          include_final_epoch=False, union_seeds=b["union"])
    else:
        ids, r, t, e = _ensemble_external(PRED / "external_preds", b["external"], b.get("ext_tag", b["tag"]), seeds, 5,
                                          include_final_epoch=False)
    return ids, r, np.asarray(t, float), np.asarray(e, int)


def cidx(kind, cohort):
    if kind == "harrell":
        return lambda r, t, e: (lambda n, d: n / d)(*harrell_c(r, t, e))
    return lambda r, t, e: uno_c(r, t, e, TAU[cohort])


def paired_cells(A, B, cohort, n_boot, rng):
    ra, rb, t, e = align(A, (B[0], B[1], B[2], B[3]))
    t = t.astype(float)
    cells, ha, hb = [], None, None
    for kind in ("harrell", "uno"):
        f = cidx(kind, cohort)
        a0, b0 = f(ra, t, e), f(rb, t, e)
        if kind == "harrell":
            ha, hb = a0, b0
        ds = []
        for _ in range(n_boot):
            ix = rng.randint(0, len(t), len(t))
            if e[ix].sum():
                ds.append(f(rb[ix], t[ix], e[ix]) - f(ra[ix], t[ix], e[ix]))
        ds = np.asarray(ds)
        lo, hi = np.percentile(ds, [2.5, 97.5])
        p = min(2 * min(np.mean(ds <= 0), np.mean(ds >= 0)), 1.0)
        star = "**" if lo > 0 or hi < 0 else ""
        cells.append(f"{star}{b0 - a0:+.4f} [{lo:+.4f}, {hi:+.4f}]{star} {p:.3f}")
    return len(t), ha, hb, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="84,126,42,168,210")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    hdr = ["| 비교 | split | n | A Harrell | B Harrell | Harrell Δ [95% CI] p | Uno Δ [95% CI] p |",
           "|---|---|---|---|---|---|---|"]

    L = ["# Published baseline 비교 (새 프로토콜, 2026-10-11)", "",
         f"baseline seeds={seeds}, 우리 모델 5 seeds x 5 folds(best+final 평균). 환자 교집합 paired bootstrap "
         f"{args.n_boot}회. Uno tau PAAD 3년, BRCA 5년. PORPOISE·MMP는 임상 입력이 없어 M3(WSI+RNA)가 같은 입력 조합.",
         "모든 baseline은 원저자 공식 코드, 우리와 같은 seed별 fold 배정(5x5), fold-safe 정규화. "
         "PAAD 시스템 수준의 CPTAC 입력은 공식 입력이 없어 근사(CNV 82개 neutral, mutation 10/14개 0).",
         "표의 A = baseline(또는 패널 비교의 앞쪽 모델), B = 우리 모델, Δ = B − A.", ""]
    for b in BASELINES:
        cohort = b["cohort"]
        L += [f"## {cohort} — {b['name']} -> 우리 모델", "", f"**{b['level']}**", ""] + hdr
        rng = np.random.RandomState(0)
        for split in (("external",) if b.get("external_only") else ("internal", "external")):
            try:
                B = load_baseline(b, split, seeds)
            except (FileNotFoundError, ValueError) as ex:
                L.append(f"| — | {split} | — | 예측 없음: {ex} | | | |")
                continue
            for name in b["compare"]:
                if name not in OURS[cohort]:
                    L.append(f"| {b['name']} -> {name} | {split} | — | 우리 모델 예측 없음(HPC 결과 대기) | | | |")
                    continue
                O = load(cohort, split, OURS[cohort][name])
                n, ha, hb, cells = paired_cells(B, O, cohort, args.n_boot, rng)
                L.append(f"| {b['name']} -> {name} | {split} | {n} | {ha:.4f} | {hb:.4f} | {cells[0]} | {cells[1]} |")
                print(f"{b['tag']} {split} ->{name} n={n} {cells[0]}", flush=True)
        L.append("")

    if PANEL_PAIRS:
      L += ["## 같은 아키텍처에서 패널만 교체 (우리 패널의 기여)", "",
          "BRCA: Hallmark(MMP 유전자 세트) -> CONS882(우리 문헌 기반 패널). Δ > 0이면 우리 패널이 더 좋음.", ""] + hdr
    rng = np.random.RandomState(1)
    for cohort, a, bname in PANEL_PAIRS:
        for split in ("internal", "external"):
            if a not in OURS[cohort]:
                L.append(f"| {a} -> {bname} | {split} | — | 예측 없음(HPC 결과 대기) | | | |")
                continue
            A = load(cohort, split, OURS[cohort][a])
            B = load(cohort, split, OURS[cohort][bname])
            n, ha, hb, cells = paired_cells(A, B, cohort, args.n_boot, rng)
            L.append(f"| {a} -> {bname} | {split} | {n} | {ha:.4f} | {hb:.4f} | {cells[0]} | {cells[1]} |")
    L.append("")
    (_ROOT / "paper" / "results_baselines.md").write_text("\n".join(L), encoding="utf-8")
    print("written paper/results_baselines.md")


if __name__ == "__main__":
    main()
