"""
PMA(WSI+RNA+age/sex, margin 없음, seed84 fold0, leakage-free fdr0.1_tcga_only) 체크포인트로
"clinical(age/sex만) 브랜치가 실제로 얼마나 기여하는가"를 측정한다 — scripts/
diagnose_pma_r_only_reliance.py와 동일한 방법(zero/permutation ablation, risk_head만 재실행),
margin 대신 age/sex를 쓴 버전과 비교하기 위한 대응 스크립트.

사용법:
    python -m scripts.diagnose_pma_agesex_reliance
"""
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, resolve_tcga_only_rna_genes
from models import ViT_PMA
from models.clinical_encoder import age_stats_from_csv
from models.rna_predictor import RNAPredictionHead
from utils.metrics import compute_survival_metrics

N_PERM_TRIALS = 20
FOLD, N_FOLDS = 0, 5
CKPT_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "checkpoint"
    / f"survival_tcga_seed84_EXTfdr0.1_SS_AUX_PMA_EXTfdr0.1_SS_AUX_DISP_FOLD{FOLD}OF{N_FOLDS}_best_pma.pt"
)
BRANCHES = ["wsi", "clinical", "rna"]


def _c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    comparable = (time[:, None] < time[None, :]) & event[:, None].astype(bool)
    concordant = comparable & (risk[:, None] > risk[None, :])
    tied = comparable & (risk[:, None] == risk[None, :])
    n = int(comparable.sum())
    return float((concordant.sum() + 0.5 * tied.sum()) / n) if n > 0 else float("nan")


@torch.no_grad()
def _patient_forward(model, patient_slides, device):
    rna = patient_slides[0]["rna"].to(device, non_blocking=True)
    z_rna = model.encode_rna(rna)  # (rna_dim,)

    components_per_slide, spatial_feats_per_slide = [], []
    for slide in patient_slides:
        coords = slide["coords"].to(device, non_blocking=True)
        out = model(coords, features=slide["features"])
        components_per_slide.append(out["embed"])  # (4, D)
        if "spatial_feat" in out:
            spatial_feats_per_slide.append(out["spatial_feat"])
    patient_components = torch.stack(components_per_slide).mean(dim=0)  # (4, D)
    spatial_feat = (
        torch.stack(spatial_feats_per_slide).mean(dim=0) if spatial_feats_per_slide else None
    )

    age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
    sex_idx = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
    z_clinical = model.clinical_encoder(age_years.unsqueeze(0), sex_idx.unsqueeze(0)).squeeze(0)

    z_wsi, coattn_weights = model.component_coattn(patient_components, z_rna)  # (D,), (4,)

    return {
        "z_wsi": z_wsi.float().cpu(),
        "z_clinical": z_clinical.float().cpu(),
        "z_rna": z_rna.float().cpu(),
        "spatial_feat": spatial_feat.float().cpu(),
        "case_id": patient_slides[0]["case_id"],
        "time": float(patient_slides[0]["OS_time"].item()),
        "event": int(patient_slides[0]["OS_event"].item()),
    }


def _collect(model, dataset, device) -> list[dict]:
    return [_patient_forward(model, dataset[i], device) for i in range(len(dataset))]


@torch.no_grad()
def _ablation_report(model, records: list[dict], device, rng: np.random.Generator) -> dict:
    parts = {
        "wsi": torch.stack([r["z_wsi"] for r in records]).to(device),
        "clinical": torch.stack([r["z_clinical"] for r in records]).to(device),
        "rna": torch.stack([r["z_rna"] for r in records]).to(device),
    }
    spatial_feat = torch.stack([r["spatial_feat"] for r in records]).to(device)
    times = np.array([r["time"] for r in records])
    events = np.array([r["event"] for r in records])
    n = len(records)

    def _batch_risk(p):
        combined = torch.cat([p["wsi"], p["clinical"], p["rna"], spatial_feat], dim=-1)
        return model.risk_head(combined).view(-1).cpu().numpy()

    baseline_risk = _batch_risk(parts)
    baseline_metrics = compute_survival_metrics(baseline_risk, times, events)

    branch_reports = {}
    for branch in BRANCHES:
        zero_parts = dict(parts)
        zero_parts[branch] = torch.zeros_like(parts[branch])
        zero_metrics = compute_survival_metrics(_batch_risk(zero_parts), times, events)

        perm_cs = []
        for _ in range(N_PERM_TRIALS):
            perm = rng.permutation(n)
            perm_parts = dict(parts)
            perm_parts[branch] = parts[branch][perm]
            perm_cs.append(_c_index(_batch_risk(perm_parts), times, events))

        branch_reports[branch] = {
            "zero_c": zero_metrics["c_index"],
            "perm_c_mean": float(np.mean(perm_cs)),
            "perm_c_std": float(np.std(perm_cs)),
        }

    return {"n": n, "baseline_c": baseline_metrics["c_index"], "baseline_hr": baseline_metrics["hr"],
            "baseline_p": baseline_metrics["log_rank_p"], "branches": branch_reports}


def _print_report(label: str, rep: dict):
    print(f"\n--- {label} (n={rep['n']}) ---")
    print(f"  baseline           : C={rep['baseline_c']:.4f}  HR={rep['baseline_hr']:.3f}  logrank_p={rep['baseline_p']:.4f}")
    for branch in BRANCHES:
        br = rep["branches"][branch]
        print(f"  [{branch:8s}] zero-ablation : C={br['zero_c']:.4f}  (baseline 대비 {br['zero_c']-rep['baseline_c']:+.4f})")
        print(f"  [{branch:8s}] perm-ablation : C={br['perm_c_mean']:.4f} +/- {br['perm_c_std']:.4f}  "
              f"(baseline 대비 {br['perm_c_mean']-rep['baseline_c']:+.4f})")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    cfg = Config()
    cfg.model.use_attn_dispersion = True
    rna_gene_ids = resolve_tcga_only_rna_genes("literature_fdr0.1_tcga_only")
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])

    print(f"RNA 유전자 수: {len(rna_gene_ids)}")
    print("데이터셋 준비 중...")
    internal_ds = WSISurvivalDataset(
        cfg.data, dataset="tcga", split="test", fold=FOLD, n_folds=N_FOLDS,
        with_clinical=True, with_rna=True, rna_gene_ids=rna_gene_ids,
    )
    external_ds = WSISurvivalDataset(
        cfg.data, dataset="cptac", split="all",
        with_clinical=True, with_rna=True, rna_gene_ids=rna_gene_ids,
    )
    print(f"internal(tcga fold{FOLD} held-out test) 환자 수: {len(internal_ds)}")
    print(f"external(cptac 전체) 환자 수: {len(external_ds)}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"체크포인트 없음: {CKPT_PATH}")

    model = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(rna_gene_ids)).to(device)
    model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, len(rna_gene_ids)).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    internal_records = _collect(model, internal_ds, device)
    external_records = _collect(model, external_ds, device)

    internal_rep = _ablation_report(model, internal_records, device, rng)
    external_rep = _ablation_report(model, external_records, device, rng)

    _print_report(f"internal(tcga fold{FOLD} held-out)", internal_rep)
    _print_report("external(cptac)", external_rep)


if __name__ == "__main__":
    main()
