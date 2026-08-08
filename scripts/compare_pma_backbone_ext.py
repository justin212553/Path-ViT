"""
PMA_INT1500_SS_AUX_R_DISP_COX_ADD를 ResNet50 vs UNI backbone으로 직접 비교한다 — 이미 두
backbone 다 이 정확한 레시피(INT1500 유전자셋, margin(R), cox_add, staging 없음)로 5-fold
체크포인트가 저장돼 있는데, external(ensemble mean) 지표만 로그가 안 남아있어 재계산한다.
internal(pooled OOF)은 이미 .logs/kfold_preds/*.csv로 저장돼 있어 pool_kfold_preds.py로 바로
확인 가능 — 이 스크립트는 external만 담당한다.

사용법: python -m scripts.compare_pma_backbone_ext
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids_intersection
from models import ViT_PMA
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv
from models.rna_predictor import RNAPredictionHead
from utils.metrics import compute_survival_metrics

N_FOLDS = 5
SEED = 84
CKPT_DIR = _ROOT / "models" / "checkpoint"


def _identity_collate(batch):
    return batch[0]


@torch.no_grad()
def external_ensemble_risk(device, backbone: str, ckpt_prefix: str):
    cfg = Config()
    cfg.model.use_attn_dispersion = True
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS["tcga"])

    ds = WSISurvivalDataset(cfg.data, dataset="cptac", split="all", with_clinical=True, with_margin=True,
                             with_staging=False, with_rna=True, rna_gene_ids=rna_gene_ids,
                             feature_backbone=backbone)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    patients = list(loader)

    fold_risks = []
    case_ids_ref, time_ref, event_ref = None, None, None
    for fold in range(N_FOLDS):
        ckpt_path = CKPT_DIR / f"{ckpt_prefix}_FOLD{fold}OF{N_FOLDS}_best_pma.pt"
        model = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(rna_gene_ids),
                         backbone=backbone, combine_mode="cox_add", use_margin=True, margin_stats=margin_stats,
                         use_age_sex=True).to(device)
        model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, len(rna_gene_ids)).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        risks, case_ids, times, events = [], [], [], []
        for patient_slides in patients:
            p = patient_slides[0]
            rna = p["rna"].to(device)
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
            age_years = p["age_years"].to(device)
            sex_idx = p["sex_idx"].to(device)
            margin_ord = p["margin_ord"].to(device)
            clin_raw = model._clinical_raw(age_years, sex_idx, margin_ord, stage_ord=None)
            risk = risk + model.clinical_linear(clin_raw).view(1)
            risks.append(risk.item())
            case_ids.append(p["case_id"])
            times.append(float(p["OS_time"].item()))
            events.append(int(p["OS_event"].item()))
        if case_ids_ref is None:
            case_ids_ref, time_ref, event_ref = case_ids, np.array(times), np.array(events)
        fold_risks.append(np.array(risks))
        c = compute_survival_metrics(np.array(risks), time_ref, event_ref)["c_index"]
        print(f"  [{backbone}] fold {fold}: external c={c:.4f}")

    mean_risk = np.mean(np.stack(fold_risks), axis=0)
    metrics = compute_survival_metrics(mean_risk, time_ref, event_ref)
    print(f"  [{backbone}] external(ensemble mean): C={metrics['c_index']:.4f}  HR={metrics['hr']:.3f}  "
          f"logrank_p={metrics['log_rank_p']:.4f}")
    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== ResNet50: PMA_INT1500_SS_AUX_R_DISP_COX_ADD ===")
    external_ensemble_risk(
        device, backbone="resnet50",
        ckpt_prefix="survival_tcga_seed84_INT1500_SS_AUX_R_PMA_INT1500_SS_AUX_R_DISP_COX_ADD",
    )

    print("\n=== UNI: PMA_uni_INT1500_SS_AUX_R_DISP_COX_ADD ===")
    external_ensemble_risk(
        device, backbone="uni",
        ckpt_prefix="survival_tcga_uni_seed84_INT1500_SS_AUX_R_PMA_uni_INT1500_SS_AUX_R_DISP_COX_ADD",
    )


if __name__ == "__main__":
    main()
