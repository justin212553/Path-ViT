"""
PMA_uni_INT1500_SS_AUX_STG_R_DISP_COX_ADD(seed84, 5-fold)의 internal(pooled OOF, N=152)과
external(ensemble mean, N=144) c-index 격차가, 각 코호트 크기에서 기대되는 표본 노이즈보다
큰 "진짜" 차이인지 bootstrap으로 확인한다.

1) fold-level 비교: 같은 fold 체크포인트가 자기 test(held-out, ~30명)와 external(144명)에서
   각각 얼마나 나오는지 나란히 놓고, 몇 번이나 internal이 external을 이기는지 센다.
2) internal(pooled, N=152) bootstrap: 152명을 복원추출로 리샘플해 c-index 분포/95% CI를 얻는다.
3) external(ensemble mean, N=144) bootstrap: 5-fold 체크포인트로 CPTAC 전체를 평가해 평균
   risk를 만든 뒤, 144명을 복원추출로 리샘플해 c-index 분포/95% CI를 얻는다.
4) 두 CI가 겹치는지로 "internal<external이 노이즈로 설명되는가"를 판단한다.

사용법: python -m scripts.bootstrap_pma_internal_external_gap
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
from models import ViT_PMA
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_csv, STAGE_FIELDS
from models.rna_predictor import RNAPredictionHead

N_FOLDS = 5
SEED = 84
N_BOOT = 2000
RNG = np.random.default_rng(84)
MODEL_TAG = "PMA_uni_INT1500_SS_AUX_STG_R_DISP_COX_ADD"
KFOLD_PREDS_DIR = _ROOT / ".logs" / "kfold_preds"
CKPT_DIR = _ROOT / "models" / "checkpoint"


def _c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    comparable = (time[:, None] < time[None, :]) & event[:, None].astype(bool)
    concordant = comparable & (risk[:, None] > risk[None, :])
    tied = comparable & (risk[:, None] == risk[None, :])
    n = int(comparable.sum())
    return float((concordant.sum() + 0.5 * tied.sum()) / n) if n > 0 else float("nan")


def _bootstrap_ci(risk: np.ndarray, time: np.ndarray, event: np.ndarray, n_boot: int = N_BOOT):
    n = len(risk)
    boot_cs = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        c = _c_index(risk[idx], time[idx], event[idx])
        if not np.isnan(c):
            boot_cs.append(c)
    boot_cs = np.array(boot_cs)
    return {
        "point": _c_index(risk, time, event),
        "boot_mean": float(boot_cs.mean()), "boot_std": float(boot_cs.std()),
        "ci_lo": float(np.percentile(boot_cs, 2.5)), "ci_hi": float(np.percentile(boot_cs, 97.5)),
    }


def _identity_collate(batch):
    return batch[0]


@torch.no_grad()
def external_ensemble_risk(device):
    cfg = Config()
    cfg.model.use_attn_dispersion = True
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS["tcga"])
    stage_stats = stage_stats_from_csv(CLINICAL_PATHS["tcga"])

    ds = WSISurvivalDataset(cfg.data, dataset="cptac", split="all", with_clinical=True, with_margin=True,
                             with_staging=True, with_rna=True, rna_gene_ids=rna_gene_ids, feature_backbone="uni")
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    patients = list(loader)

    fold_risks = []
    case_ids_ref, time_ref, event_ref = None, None, None
    for fold in range(N_FOLDS):
        ckpt_path = CKPT_DIR / (
            f"survival_tcga_uni_seed{SEED}_INT1500_SS_AUX_STG_R_PMA_uni_INT1500_SS_AUX_STG_R_DISP_COX_ADD_"
            f"FOLD{fold}OF{N_FOLDS}_best_pma.pt"
        )
        model = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(rna_gene_ids),
                         backbone="uni", combine_mode="cox_add", use_margin=True, margin_stats=margin_stats,
                         use_age_sex=True, use_staging=True, stage_stats=stage_stats).to(device)
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
            stage_ord = {f: p[f].to(device) for f in STAGE_FIELDS}
            clin_raw = model._clinical_raw(age_years, sex_idx, margin_ord, stage_ord=stage_ord)
            risk = risk + model.clinical_linear(clin_raw).view(1)
            risks.append(risk.item())
            case_ids.append(p["case_id"])
            times.append(float(p["OS_time"].item()))
            events.append(int(p["OS_event"].item()))
        if case_ids_ref is None:
            case_ids_ref, time_ref, event_ref = case_ids, np.array(times), np.array(events)
        fold_risks.append(np.array(risks))
        print(f"  fold {fold}: external c={_c_index(np.array(risks), time_ref, event_ref):.4f}")

    mean_risk = np.mean(np.stack(fold_risks), axis=0)
    return mean_risk, time_ref, event_ref


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== (1) Internal 로드 (pooled OOF, N=152) ===")
    dfs = [pd.read_csv(KFOLD_PREDS_DIR / f"tcga_{MODEL_TAG}_FOLD{f}OF{N_FOLDS}_seed{SEED}_fold{f}of{N_FOLDS}.csv")
           for f in range(N_FOLDS)]
    internal_df = pd.concat(dfs, ignore_index=True)
    internal_ci = _bootstrap_ci(internal_df["risk"].values, internal_df["OS_time"].values, internal_df["OS_event"].values)
    print(f"  internal point={internal_ci['point']:.4f}  bootstrap 95%CI=[{internal_ci['ci_lo']:.4f}, {internal_ci['ci_hi']:.4f}]  "
          f"(boot_std={internal_ci['boot_std']:.4f})")

    print("\n=== (2) External ensemble risk 계산 (5-fold 체크포인트 각각 forward) ===")
    ext_risk, ext_time, ext_event = external_ensemble_risk(device)
    external_ci = _bootstrap_ci(ext_risk, ext_time, ext_event)
    print(f"  external point={external_ci['point']:.4f}  bootstrap 95%CI=[{external_ci['ci_lo']:.4f}, {external_ci['ci_hi']:.4f}]  "
          f"(boot_std={external_ci['boot_std']:.4f})")

    print(f"\n=== (3) 두 CI 겹침 여부 ===")
    overlap = not (internal_ci["ci_hi"] < external_ci["ci_lo"] or external_ci["ci_hi"] < internal_ci["ci_lo"])
    print(f"  internal 95%CI: [{internal_ci['ci_lo']:.4f}, {internal_ci['ci_hi']:.4f}]")
    print(f"  external 95%CI: [{external_ci['ci_lo']:.4f}, {external_ci['ci_hi']:.4f}]")
    print(f"  겹침: {overlap}")

    print(f"\n=== (4) Bootstrap 차이 분포로 직접 검정 (external - internal) ===")
    n_int, n_ext = len(internal_df), len(ext_risk)
    diffs = []
    for _ in range(N_BOOT):
        idx_i = RNG.integers(0, n_int, size=n_int)
        idx_e = RNG.integers(0, n_ext, size=n_ext)
        c_i = _c_index(internal_df["risk"].values[idx_i], internal_df["OS_time"].values[idx_i], internal_df["OS_event"].values[idx_i])
        c_e = _c_index(ext_risk[idx_e], ext_time[idx_e], ext_event[idx_e])
        if not (np.isnan(c_i) or np.isnan(c_e)):
            diffs.append(c_e - c_i)
    diffs = np.array(diffs)
    p_internal_ge_external = float((diffs <= 0).mean())  # external-internal<=0, 즉 internal>=external인 비율
    print(f"  (external - internal) bootstrap 분포: mean={diffs.mean():.4f}, 95%CI=[{np.percentile(diffs,2.5):.4f}, {np.percentile(diffs,97.5):.4f}]")
    print(f"  P(internal >= external) = {p_internal_ge_external:.4f} (독립 리샘플링 기준 — 낮을수록 'external>internal'이 노이즈로 설명 안 됨)")


if __name__ == "__main__":
    main()
