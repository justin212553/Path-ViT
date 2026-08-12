"""
M7(Clinical+RNA, cox_add, WSI 없음) vs PMA(Clinical+RNA+WSI, cox_add) — 두 모델이 "맞추는" 환자와
"틀리는" 환자가 얼마나 겹치는지 확인한다. WSI 하나만 더해졌을 뿐인데 M7이 PMA보다 나은 지표를
보이는 지점(M7 internal=0.6373 vs PMA internal=0.6359, 반대로 external은 PMA=0.6337 > M7=0.6216)이
있어, "WSI를 더하면 어떤 환자에서 이득/손해를 보는가"를 환자 단위로 뜯어본다.

환자 단위 "맞춤/틀림"은 개별 라벨이 없는 survival 세팅이라 pairwise concordance로 정의한다
(audit_leverage_patients.py와 동일한 comparable/concordant/tied 정의, Harrell's c-index 기준):
그 환자가 관여하는 모든 comparable pair 중 risk 순서가 실제 순서와 어긋난 비율
(discordant fraction). 0.5 미만이면 "이 환자에 관해서는 모델이 평균적으로 맞게 정렬한다" ->
"맞춘 환자", 0.5 이상이면 "틀린 환자"로 이분화한다.

internal은 시드별 5-fold 풀링 OOF(각 환자가 held-out으로 정확히 1번 예측됨), external은
시드별 5-fold 체크포인트 예측 평균을 사용해 "시드별로" 나눠서 보고, 3시드 전체 풀링 결과도 같이
보여준다(사용자 요청: "시드별로 맞춘거 틀린거를 나눠보고 확인해봐").

사용법: python scripts/compare_m7_pma_agreement.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.metrics import compute_survival_metrics

SEEDS = [42, 84, 126]
N_FOLDS = 5

MODEL_M7 = "M7_INT1500_STG_R_COX_ADD"
MODEL_PMA = "PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD"

GOOD_THRESHOLD = 0.5  # discordant fraction 이 값 미만이면 "맞춘 환자"


def _load_csv(path: Path) -> dict:
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["case_id"]] = (float(row["risk"]), float(row["OS_time"]), int(row["OS_event"]))
    return out


def _internal_seed(dataset: str, model: str, seed: int) -> dict:
    """5-fold 풀링 OOF — 환자당 정확히 1개 예측(자신을 held-out한 fold)."""
    pred_dir = _ROOT / ".logs" / "kfold_preds"
    preds = {}
    for fold in range(N_FOLDS):
        path = pred_dir / f"{dataset}_{model}_FOLD{fold}OF{N_FOLDS}_seed{seed}_fold{fold}of{N_FOLDS}.csv"
        if not path.exists():
            path = pred_dir / f"{dataset}_{model}_seed{seed}_fold{fold}of{N_FOLDS}.csv"
        preds.update(_load_csv(path))
    return preds


def _external_seed(dataset: str, model: str, seed: int) -> dict:
    """같은 external 환자 144명을 그 시드의 5-fold 체크포인트 예측 평균으로."""
    pred_dir = _ROOT / ".logs" / "external_preds"
    risks, label = {}, {}
    for fold in range(N_FOLDS):
        path = pred_dir / f"{dataset}_{model}_FOLD{fold}OF{N_FOLDS}_seed{seed}_fold{fold}of{N_FOLDS}.csv"
        if not path.exists():
            path = pred_dir / f"{dataset}_{model}_seed{seed}_fold{fold}of{N_FOLDS}.csv"
        for cid, (r, t, e) in _load_csv(path).items():
            risks.setdefault(cid, []).append(r)
            label[cid] = (t, e)
    return {cid: (float(np.mean(rs)), *label[cid]) for cid, rs in risks.items()}


def _pool_seeds(per_seed: dict) -> dict:
    """여러 시드 dict(각 case_id -> (risk,time,event))를 환자별 risk 평균으로 풀링."""
    seeds = list(per_seed.keys())
    common = sorted(set.intersection(*(set(per_seed[s].keys()) for s in seeds)))
    out = {}
    for cid in common:
        rs = [per_seed[s][cid][0] for s in seeds]
        _, t, e = per_seed[seeds[0]][cid]
        out[cid] = (float(np.mean(rs)), t, e)
    return out


def _load_clinical(dataset: str) -> dict:
    path = _ROOT / "data" / f"clinical_{dataset}.csv"
    out = {}
    if not path.exists():
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["case_id"]] = {
                "ajcc_stage": row.get("ajcc_stage", ""),
                "tumor_grade": row.get("tumor_grade", ""),
            }
    return out


def _disc_frac(risks: np.ndarray, times: np.ndarray, events: np.ndarray):
    comparable = (times[:, None] < times[None, :]) & events[:, None]
    concordant = comparable & (risks[:, None] > risks[None, :])
    tied = comparable & (risks[:, None] == risks[None, :])
    row_comp, col_comp = comparable.sum(1), comparable.sum(0)
    row_conc, col_conc = concordant.sum(1), concordant.sum(0)
    row_tied, col_tied = tied.sum(1), tied.sum(0)
    involvement = row_comp + col_comp
    total_disc = (row_comp - row_conc - row_tied) + (col_comp - col_conc - col_tied)
    with np.errstate(invalid="ignore", divide="ignore"):
        disc_frac = np.where(involvement > 0, total_disc / involvement, np.nan)
    return disc_frac, involvement


def _analyze(label: str, dataset: str, preds_m7: dict, preds_pma: dict, clinical: dict, top_n: int = 8):
    common = sorted(set(preds_m7.keys()) & set(preds_pma.keys()))
    times = np.array([preds_m7[c][1] for c in common])
    events = np.array([preds_m7[c][2] for c in common], dtype=bool)
    r_m7 = np.array([preds_m7[c][0] for c in common])
    r_pma = np.array([preds_pma[c][0] for c in common])

    c_m7 = compute_survival_metrics(r_m7, times, events)["c_index"]
    c_pma = compute_survival_metrics(r_pma, times, events)["c_index"]

    disc_m7, inv_m7 = _disc_frac(r_m7, times, events)
    disc_pma, inv_pma = _disc_frac(r_pma, times, events)

    valid = (inv_m7 > 0) & (inv_pma > 0)
    n_valid = int(valid.sum())

    good_m7 = disc_m7 < GOOD_THRESHOLD
    good_pma = disc_pma < GOOD_THRESHOLD

    gg = int((good_m7 & good_pma & valid).sum())
    gb = int((good_m7 & ~good_pma & valid).sum())  # M7만 맞춤
    bg = int((~good_m7 & good_pma & valid).sum())  # PMA만 맞춤
    bb = int((~good_m7 & ~good_pma & valid).sum())

    rho, _ = spearmanr(disc_m7[valid], disc_pma[valid])
    jaccard_bad = (int((~good_m7 & ~good_pma & valid).sum())
                   / max(1, int(((~good_m7 | ~good_pma) & valid).sum())))

    print(f"\n=== [{label}] {dataset} N={n_valid} (M7 c={c_m7:.4f}, PMA c={c_pma:.4f}) ===")
    print(f"  둘 다 맞춤(good-good)     : {gg:4d} ({gg/n_valid*100:5.1f}%)")
    print(f"  M7만 맞춤(good-bad)       : {gb:4d} ({gb/n_valid*100:5.1f}%)")
    print(f"  PMA만 맞춤(bad-good)      : {bg:4d} ({bg/n_valid*100:5.1f}%)")
    print(f"  둘 다 틀림(bad-bad)       : {bb:4d} ({bb/n_valid*100:5.1f}%)")
    print(f"  disc_frac Spearman rho={rho:.3f} | '틀린 환자' Jaccard overlap={jaccard_bad:.3f}")

    def fmt(idx):
        cid = common[idx]
        cl = clinical.get(cid, {})
        return (f"{cid:15s} time={times[idx]:6.0f} event={int(events[idx])} "
                f"M7_disc={disc_m7[idx]:.3f}(inv={inv_m7[idx]:.0f}) "
                f"PMA_disc={disc_pma[idx]:.3f}(inv={inv_pma[idx]:.0f}) "
                f"stage={cl.get('ajcc_stage','?'):10s} grade={cl.get('tumor_grade','?')}")

    diff = disc_pma - disc_m7  # 양수 클수록 "M7은 맞추는데 PMA는 틀림"
    cand = np.where(valid & good_m7 & ~good_pma)[0]
    cand = cand[np.argsort(-diff[cand])]
    print(f"  --- M7은 맞추는데 PMA는 틀리는 환자 (상위 {top_n}) ---")
    for idx in cand[:top_n]:
        print("    " + fmt(idx))

    cand2 = np.where(valid & good_pma & ~good_m7)[0]
    cand2 = cand2[np.argsort(diff[cand2])]
    print(f"  --- PMA는 맞추는데 M7은 틀리는 환자 (상위 {top_n}) ---")
    for idx in cand2[:top_n]:
        print("    " + fmt(idx))

    return {"gg": gg, "gb": gb, "bg": bg, "bb": bb, "n": n_valid, "rho": rho, "jaccard_bad": jaccard_bad}


def main():
    clinical_tcga = _load_clinical("tcga")
    clinical_cptac = _load_clinical("cptac")

    print("########## INTERNAL (시드별) ##########")
    internal_m7_per_seed, internal_pma_per_seed = {}, {}
    for seed in SEEDS:
        m7 = _internal_seed("tcga", MODEL_M7, seed)
        pma = _internal_seed("tcga", MODEL_PMA, seed)
        internal_m7_per_seed[seed] = m7
        internal_pma_per_seed[seed] = pma
        _analyze(f"internal seed={seed}", "tcga", m7, pma, clinical_tcga)

    print("\n########## INTERNAL (3시드 풀링) ##########")
    _analyze("internal pooled(3seed)", "tcga",
              _pool_seeds(internal_m7_per_seed), _pool_seeds(internal_pma_per_seed), clinical_tcga)

    print("\n\n########## EXTERNAL (시드별) ##########")
    external_m7_per_seed, external_pma_per_seed = {}, {}
    for seed in SEEDS:
        m7 = _external_seed("cptac", MODEL_M7, seed)
        pma = _external_seed("cptac", MODEL_PMA, seed)
        external_m7_per_seed[seed] = m7
        external_pma_per_seed[seed] = pma
        _analyze(f"external seed={seed}", "cptac", m7, pma, clinical_cptac)

    print("\n########## EXTERNAL (3시드 풀링, 15실행 평균) ##########")
    _analyze("external pooled(15run)", "cptac",
              _pool_seeds(external_m7_per_seed), _pool_seeds(external_pma_per_seed), clinical_cptac)


if __name__ == "__main__":
    main()
