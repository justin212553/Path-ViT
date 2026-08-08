"""
M1_POOL(UNI, WSI 단독, 다성분 pooling+self-attention)과 M5(Clinical age/sex 단독) — 지금 가진
모델 중 각각 "WSI 단독"/"clinical 단독" 최고 성능 — 를 완전히 독립적으로 학습된 채로 두고,
최종 risk score만 Cox 선형결합(risk = α·risk_M1POOL + β·risk_M5)한다.
scripts/stack_m3_m7r_cox.py와 동일한 방법론(재학습 없이 5-fold 체크포인트의 risk score만 뽑아
작은 Cox 회귀로 결합 계수를 적합)을 M1_POOL+M5에 적용한 버전.

배경: M2_POOL(WSI+clinical을 co-attention/cox_add로 "같이" 학습)이 계속 external 0.49~0.50에
머문 반면, M1_POOL 단독은 0.556까지 나왔다 — clinical(age/sex, 약한 신호)이 WSI pooling과
"같이" 학습되면(co-attention query든 zero-init cox_add든) 오히려 방해가 된다는 관찰. 그렇다면
아예 두 모델을 따로따로 완전히 독립적으로 학습시킨 뒤, 그 다음 단계(risk score 레벨)에서만
합치면 어느 쪽 학습도 서로 간섭하지 않을 것이라는 가설을 검증한다.

[Internal] 두 모델의 5-fold OOF risk(.logs/kfold_preds/*.csv, 이미 저장돼 있음)를 case_id로
병합해 CoxPHFitter로 계수를 적합한 뒤 같은 데이터에 대해 결합 risk를 계산한다(계수를 적합한
데이터로 바로 평가하는 낙관적 편향이 있음 — 2개 파라미터뿐이라 크지 않을 것으로 보이나 명확히
표시한다).

[External] 두 모델 각각 5-fold 체크포인트로 CPTAC 전체를 평가해 risk를 얻고, fold 5개를
평균(ensemble mean)한 뒤 위에서 적합한 (α, β)로 결합한다.

사용법: python -m scripts.stack_m1pool_m5_cox
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS
from models import ViT_M1_Pool, ClinicalOnly
from models.clinical_encoder import age_stats_from_csv
from utils.metrics import compute_survival_metrics

try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

N_FOLDS = 5
SEED = 84
KFOLD_PREDS_DIR = _ROOT / ".logs" / "kfold_preds"
CKPT_DIR = _ROOT / "models" / "checkpoint"


def _identity_collate(batch):
    return batch[0]


def load_internal_oof(model_tag: str) -> pd.DataFrame:
    dfs = []
    for fold in range(N_FOLDS):
        path = KFOLD_PREDS_DIR / f"tcga_{model_tag}_FOLD{fold}OF{N_FOLDS}_seed{SEED}_fold{fold}of{N_FOLDS}.csv"
        dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)


@torch.no_grad()
def m1pool_external_risks(device) -> dict[str, np.ndarray]:
    """M1_POOL(UNI, self-attention, clinical 없음) 5-fold 체크포인트 각각으로 CPTAC 전체를 평가한다."""
    cfg = Config()
    ds = WSISurvivalDataset(cfg.data, dataset="cptac", split="all", with_clinical=False, with_rna=False,
                             feature_backbone="uni")
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    patients = list(loader)

    fold_risks = []
    case_ids_ref, time_ref, event_ref = None, None, None
    for fold in range(N_FOLDS):
        ckpt_path = CKPT_DIR / f"survival_tcga_uni_seed{SEED}_SS_M1_POOL_uni_SS_DISP_FOLD{fold}OF{N_FOLDS}_best_m1_pool.pt"
        model = ViT_M1_Pool(cfg.model, backbone="uni", use_attn_dispersion=True).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        risks, case_ids, times, events = [], [], [], []
        for patient_slides in patients:
            comps, spatial_feats = [], []
            for slide in patient_slides:
                out = model(slide["coords"].to(device), features=slide["features"])
                comps.append(out["embed"])
                if "spatial_feat" in out:
                    spatial_feats.append(out["spatial_feat"])
            patient_embed = torch.stack(comps).mean(dim=0)  # (4, D)
            spatial_feat = torch.stack(spatial_feats).mean(dim=0) if spatial_feats else None
            z_wsi = model.pool_components(patient_embed)
            fused = z_wsi if spatial_feat is None else torch.cat([z_wsi, spatial_feat], dim=-1)
            risk = model.risk_head(fused.unsqueeze(0)).view(1)
            risks.append(risk.item())
            case_ids.append(patient_slides[0]["case_id"])
            times.append(float(patient_slides[0]["OS_time"].item()))
            events.append(int(patient_slides[0]["OS_event"].item()))
        if case_ids_ref is None:
            case_ids_ref, time_ref, event_ref = case_ids, np.array(times), np.array(events)
        fold_risks.append(np.array(risks))
        print(f"  M1_POOL external fold {fold}: c={compute_survival_metrics(np.array(risks), time_ref, event_ref)['c_index']:.4f}")

    mean_risk = np.mean(np.stack(fold_risks), axis=0)
    return {"case_id": case_ids_ref, "risk": mean_risk, "time": time_ref, "event": event_ref}


@torch.no_grad()
def m5_external_risks(device) -> dict[str, np.ndarray]:
    """M5(ClinicalOnly, age/sex만) 5-fold 체크포인트로 CPTAC 전체를 평가한다."""
    cfg = Config()
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])
    ds = WSISurvivalDataset(cfg.data, dataset="cptac", split="all", with_clinical=True, with_rna=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    patients = list(loader)

    fold_risks = []
    case_ids_ref, time_ref, event_ref = None, None, None
    for fold in range(N_FOLDS):
        ckpt_path = CKPT_DIR / f"survival_tcga_best_m5_fold{fold}of{N_FOLDS}_seed{SEED}_light.pt"
        model = ClinicalOnly(cfg.model, age_mean=age_mean, age_std=age_std).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        risks, case_ids, times, events = [], [], [], []
        for patient_slides in patients:
            p = patient_slides[0]
            risk = model(p["age_years"].to(device), p["sex_idx"].to(device))
            risks.append(risk.item())
            case_ids.append(p["case_id"])
            times.append(float(p["OS_time"].item()))
            events.append(int(p["OS_event"].item()))
        if case_ids_ref is None:
            case_ids_ref, time_ref, event_ref = case_ids, np.array(times), np.array(events)
        fold_risks.append(np.array(risks))
        print(f"  M5 external fold {fold}: c={compute_survival_metrics(np.array(risks), time_ref, event_ref)['c_index']:.4f}")

    mean_risk = np.mean(np.stack(fold_risks), axis=0)
    return {"case_id": case_ids_ref, "risk": mean_risk, "time": time_ref, "event": event_ref}


def main():
    if not LIFELINES_AVAILABLE:
        raise RuntimeError("lifelines가 필요합니다.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Internal OOF risk 로드 ===")
    m1pool_internal = load_internal_oof("M1_POOL_uni_SS_DISP")[["case_id", "risk", "OS_time", "OS_event"]]
    m1pool_internal = m1pool_internal.rename(columns={"risk": "risk_m1pool"})
    m5_internal = load_internal_oof("M5")[["case_id", "risk"]]
    m5_internal = m5_internal.rename(columns={"risk": "risk_m5"})
    merged = m1pool_internal.merge(m5_internal, on="case_id", how="inner")
    print(f"내부 병합 환자 수: {len(merged)} (M1_POOL={len(m1pool_internal)}, M5={len(m5_internal)})")

    baseline_m1pool = compute_survival_metrics(merged["risk_m1pool"].values, merged["OS_time"].values, merged["OS_event"].values)
    baseline_m5 = compute_survival_metrics(merged["risk_m5"].values, merged["OS_time"].values, merged["OS_event"].values)
    print(f"  (참고) M1_POOL 단독 internal: C={baseline_m1pool['c_index']:.4f}")
    print(f"  (참고) M5 단독 internal: C={baseline_m5['c_index']:.4f}")

    print("\n=== Cox 결합 계수(α, β) 적합 (internal OOF risk 전체 사용) ===")
    cph = CoxPHFitter()
    cph.fit(
        pd.DataFrame({"time": merged["OS_time"], "event": merged["OS_event"],
                      "risk_m1pool": merged["risk_m1pool"], "risk_m5": merged["risk_m5"]}),
        duration_col="time", event_col="event",
    )
    alpha = float(cph.params_["risk_m1pool"])
    beta = float(cph.params_["risk_m5"])
    print(f"  alpha(risk_m1pool 계수)={alpha:.4f}  beta(risk_m5 계수)={beta:.4f}")

    combined_internal_risk = alpha * merged["risk_m1pool"].values + beta * merged["risk_m5"].values
    combined_internal_metrics = compute_survival_metrics(combined_internal_risk, merged["OS_time"].values, merged["OS_event"].values)
    print(f"\n  결합 internal(같은 데이터로 계수 적합 + 평가라 낙관적 편향 있음 주의): "
          f"C={combined_internal_metrics['c_index']:.4f}  HR={combined_internal_metrics['hr']:.3f}  "
          f"logrank_p={combined_internal_metrics['log_rank_p']:.4f}")

    print("\n=== External risk 계산 (5-fold 체크포인트 각각 forward, ensemble mean) ===")
    print("M1_POOL(UNI, WSI 단독, self-attention):")
    m1pool_ext = m1pool_external_risks(device)
    print("M5(Clinical age/sex 단독):")
    m5_ext = m5_external_risks(device)

    ext_df = pd.DataFrame({"case_id": m1pool_ext["case_id"], "risk_m1pool": m1pool_ext["risk"],
                            "time": m1pool_ext["time"], "event": m1pool_ext["event"]}).merge(
        pd.DataFrame({"case_id": m5_ext["case_id"], "risk_m5": m5_ext["risk"]}), on="case_id", how="inner"
    )
    print(f"external 병합 환자 수: {len(ext_df)}")

    baseline_m1pool_ext = compute_survival_metrics(ext_df["risk_m1pool"].values, ext_df["time"].values, ext_df["event"].values)
    baseline_m5_ext = compute_survival_metrics(ext_df["risk_m5"].values, ext_df["time"].values, ext_df["event"].values)
    print(f"  (참고) M1_POOL 단독 external(ensemble mean): C={baseline_m1pool_ext['c_index']:.4f}")
    print(f"  (참고) M5 단독 external(ensemble mean): C={baseline_m5_ext['c_index']:.4f}")

    combined_ext_risk = alpha * ext_df["risk_m1pool"].values + beta * ext_df["risk_m5"].values
    combined_ext_metrics = compute_survival_metrics(combined_ext_risk, ext_df["time"].values, ext_df["event"].values)
    print(f"\n  결합 external: C={combined_ext_metrics['c_index']:.4f}  HR={combined_ext_metrics['hr']:.3f}  "
          f"logrank_p={combined_ext_metrics['log_rank_p']:.4f}")


if __name__ == "__main__":
    main()
