"""
M3(UNI, WSI+RNA, clinical 없음)와 M7_R(RNA+Clinical, WSI 없음) — 지금 가진 모델 중 각각
"WSI 포함"/"WSI 없음" 최고 성능 — 를 완전히 독립적으로 둔 채, 최종 risk score만
Cox 선형결합(risk = α·risk_M3 + β·risk_M7_R)한다. cox_add를 브랜치 레벨이 아니라 모델
레벨로 확장한 버전 — 재학습 없이 이미 학습된 5-fold 체크포인트의 risk score만 뽑아 결합
계수(α, β)를 작은 Cox 회귀로 새로 적합한다(스태킹 앙상블).

[Internal] 두 모델의 5-fold OOF risk(.logs/kfold_preds/*.csv, 이미 저장돼 있음)를 case_id로
병합해 152명 전체에 대해 (risk_M3, risk_M7_R) 쌍을 만들고, lifelines CoxPHFitter로 계수를
적합한 뒤 같은 152명에 대해 결합 risk를 계산한다(계수를 적합한 데이터로 바로 평가하는
낙관적 편향이 있음 — 2개 파라미터뿐이라 크지 않을 것으로 보이나, 명확히 표시한다).

[External] 두 모델 각각 5-fold 체크포인트로 CPTAC 전체를 평가해 risk를 얻고, fold 5개를
평균(ensemble mean, scripts/pool_external_preds.py와 동일 방식)한 뒤 위에서 적합한
(α, β)로 결합한다.

사용법: python -m scripts.stack_m3_m7r_cox
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
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids_intersection
from models import ViT_PMA, ClinicalRNAOnly
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv
from models.rna_predictor import RNAPredictionHead
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
def m3_external_risks(device) -> dict[str, np.ndarray]:
    """M3(ViT_PMA, UNI, clinical 없음) 5-fold 체크포인트 각각으로 CPTAC 전체를 평가한다."""
    cfg = Config()
    cfg.model.use_attn_dispersion = True
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)
    ds = WSISurvivalDataset(cfg.data, dataset="cptac", split="all", with_clinical=False, with_rna=True,
                             rna_gene_ids=rna_gene_ids, feature_backbone="uni")
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    patients = list(loader)

    fold_risks = []
    case_ids_ref, time_ref, event_ref = None, None, None
    for fold in range(N_FOLDS):
        ckpt_path = CKPT_DIR / (
            f"survival_tcga_uni_seed{SEED}_INT1500_SS_AUX_PMA_uni_INT1500_SS_AUX_NOCLINICAL_DISP_"
            f"FOLD{fold}OF{N_FOLDS}_best_pma.pt"
        )
        model = ViT_PMA(cfg.model, age_mean=None, age_std=None, rna_input_dim=len(rna_gene_ids),
                         backbone="uni", use_clinical=False).to(device)
        model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, len(rna_gene_ids)).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        risks, case_ids, times, events = [], [], [], []
        for patient_slides in patients:
            rna = patient_slides[0]["rna"].to(device)
            z_rna = model.encode_rna(rna)
            comps, spatial_feats = [], []
            for slide in patient_slides:
                out = model(slide["coords"].to(device), features=slide["features"])
                comps.append(out["embed"])
                if "spatial_feat" in out:
                    spatial_feats.append(out["spatial_feat"])
            patient_embed = torch.stack(comps).mean(dim=0)
            spatial_feat = torch.stack(spatial_feats).mean(dim=0) if spatial_feats else None
            z_wsi, _ = model.component_coattn(patient_embed, z_rna)
            fused = torch.cat([z_wsi, z_rna], dim=-1)
            if spatial_feat is not None:
                fused = torch.cat([fused, spatial_feat], dim=-1)
            risk = model.risk_head(fused.unsqueeze(0)).view(1)
            risks.append(risk.item())
            case_ids.append(patient_slides[0]["case_id"])
            times.append(float(patient_slides[0]["OS_time"].item()))
            events.append(int(patient_slides[0]["OS_event"].item()))
        if case_ids_ref is None:
            case_ids_ref, time_ref, event_ref = case_ids, np.array(times), np.array(events)
        fold_risks.append(np.array(risks))
        print(f"  M3 external fold {fold}: c={compute_survival_metrics(np.array(risks), time_ref, event_ref)['c_index']:.4f}")

    mean_risk = np.mean(np.stack(fold_risks), axis=0)
    return {"case_id": case_ids_ref, "risk": mean_risk, "time": time_ref, "event": event_ref}


@torch.no_grad()
def m7r_external_risks(device) -> dict[str, np.ndarray]:
    """M7_R(ClinicalRNAOnly, concat, clinical_dim=64/rna_dim=64) 5-fold 체크포인트로 CPTAC 전체를 평가한다."""
    cfg = Config()
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS["tcga"])
    ds = WSISurvivalDataset(cfg.data, dataset="cptac", split="all", with_clinical=True, with_margin=True,
                             with_rna=True, rna_gene_ids=rna_gene_ids)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    patients = list(loader)

    fold_risks = []
    case_ids_ref, time_ref, event_ref = None, None, None
    for fold in range(N_FOLDS):
        ckpt_path = CKPT_DIR / f"survival_tcga_best_m7_int1500_r_clindim64_rnadim64_fold{fold}of{N_FOLDS}_seed{SEED}_light.pt"
        model = ClinicalRNAOnly(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(rna_gene_ids),
                                 clinical_dim=64, rna_dim=64, combine_mode="concat",
                                 use_margin=True, margin_stats=margin_stats, use_age_sex=True).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        risks, case_ids, times, events = [], [], [], []
        for patient_slides in patients:
            p = patient_slides[0]
            risk = model(p["age_years"].to(device), p["sex_idx"].to(device), p["rna"].to(device),
                         margin_ord=p["margin_ord"].to(device))
            risks.append(risk.item())
            case_ids.append(p["case_id"])
            times.append(float(p["OS_time"].item()))
            events.append(int(p["OS_event"].item()))
        if case_ids_ref is None:
            case_ids_ref, time_ref, event_ref = case_ids, np.array(times), np.array(events)
        fold_risks.append(np.array(risks))
        print(f"  M7_R external fold {fold}: c={compute_survival_metrics(np.array(risks), time_ref, event_ref)['c_index']:.4f}")

    mean_risk = np.mean(np.stack(fold_risks), axis=0)
    return {"case_id": case_ids_ref, "risk": mean_risk, "time": time_ref, "event": event_ref}


def main():
    if not LIFELINES_AVAILABLE:
        raise RuntimeError("lifelines가 필요합니다.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Internal OOF risk 로드 ===")
    m3_internal = load_internal_oof("PMA_uni_INT1500_SS_AUX_NOCLINICAL_DISP")[["case_id", "risk", "OS_time", "OS_event"]]
    m3_internal = m3_internal.rename(columns={"risk": "risk_m3"})
    m7r_internal = load_internal_oof("M7_INT1500_R_CLINDIM64_RNADIM64")[["case_id", "risk"]]
    m7r_internal = m7r_internal.rename(columns={"risk": "risk_m7r"})
    merged = m3_internal.merge(m7r_internal, on="case_id", how="inner")
    print(f"내부 병합 환자 수: {len(merged)} (M3={len(m3_internal)}, M7_R={len(m7r_internal)})")

    baseline_m3 = compute_survival_metrics(merged["risk_m3"].values, merged["OS_time"].values, merged["OS_event"].values)
    baseline_m7r = compute_survival_metrics(merged["risk_m7r"].values, merged["OS_time"].values, merged["OS_event"].values)
    print(f"  (참고) M3 단독 internal: C={baseline_m3['c_index']:.4f}")
    print(f"  (참고) M7_R 단독 internal: C={baseline_m7r['c_index']:.4f}")

    print("\n=== Cox 결합 계수(α, β) 적합 (internal OOF risk 전체 사용) ===")
    cph = CoxPHFitter()
    cph.fit(
        pd.DataFrame({"time": merged["OS_time"], "event": merged["OS_event"],
                      "risk_m3": merged["risk_m3"], "risk_m7r": merged["risk_m7r"]}),
        duration_col="time", event_col="event",
    )
    alpha = float(cph.params_["risk_m3"])
    beta = float(cph.params_["risk_m7r"])
    print(f"  alpha(risk_m3 계수)={alpha:.4f}  beta(risk_m7r 계수)={beta:.4f}")

    combined_internal_risk = alpha * merged["risk_m3"].values + beta * merged["risk_m7r"].values
    combined_internal_metrics = compute_survival_metrics(combined_internal_risk, merged["OS_time"].values, merged["OS_event"].values)
    print(f"\n  결합 internal(같은 데이터로 계수 적합 + 평가라 낙관적 편향 있음 주의): "
          f"C={combined_internal_metrics['c_index']:.4f}  HR={combined_internal_metrics['hr']:.3f}  "
          f"logrank_p={combined_internal_metrics['log_rank_p']:.4f}")

    print("\n=== External risk 계산 (5-fold 체크포인트 각각 forward, ensemble mean) ===")
    print("M3(UNI, WSI+RNA):")
    m3_ext = m3_external_risks(device)
    print("M7_R(RNA+Clinical):")
    m7r_ext = m7r_external_risks(device)

    ext_df = pd.DataFrame({"case_id": m3_ext["case_id"], "risk_m3": m3_ext["risk"],
                            "time": m3_ext["time"], "event": m3_ext["event"]}).merge(
        pd.DataFrame({"case_id": m7r_ext["case_id"], "risk_m7r": m7r_ext["risk"]}), on="case_id", how="inner"
    )
    print(f"external 병합 환자 수: {len(ext_df)}")

    baseline_m3_ext = compute_survival_metrics(ext_df["risk_m3"].values, ext_df["time"].values, ext_df["event"].values)
    baseline_m7r_ext = compute_survival_metrics(ext_df["risk_m7r"].values, ext_df["time"].values, ext_df["event"].values)
    print(f"  (참고) M3 단독 external(ensemble mean): C={baseline_m3_ext['c_index']:.4f}")
    print(f"  (참고) M7_R 단독 external(ensemble mean): C={baseline_m7r_ext['c_index']:.4f}")

    combined_ext_risk = alpha * ext_df["risk_m3"].values + beta * ext_df["risk_m7r"].values
    combined_ext_metrics = compute_survival_metrics(combined_ext_risk, ext_df["time"].values, ext_df["event"].values)
    print(f"\n  결합 external: C={combined_ext_metrics['c_index']:.4f}  HR={combined_ext_metrics['hr']:.3f}  "
          f"logrank_p={combined_ext_metrics['log_rank_p']:.4f}")


if __name__ == "__main__":
    main()
