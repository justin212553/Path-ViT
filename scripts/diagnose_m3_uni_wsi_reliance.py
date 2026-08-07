"""
M3(PMA_uni_INT1500_SS_AUX_NOCLINICAL_DISP, seed84, 5-fold, UNI 백본) 체크포인트로
branch-ablation을 돌려, UNI로 바꾼 뒤 WSI가 실제로 risk 예측에 얼마나 기여하는지 역산한다
(scripts/diagnose_pma_uni_r_reliance.py의 M3 버전 — clinical 브랜치가 없어 wsi/rna 두 개만 본다).

ResNet50-SwAV 시절 진단(scripts/diagnose_wsi_reliance.py)에서는 WSI 기여도가 거의 0이었다 —
UNI로 바꾼 뒤 external c-index가 크게 오른 게(0.567->0.638) 실제로 WSI 브랜치 자체의 기여가
커져서인지, 아니면 다른 간접 효과(정규화 등)인지 직접 ablation으로 확인한다.

사용법: python -m scripts.diagnose_m3_uni_wsi_reliance
"""
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, literature_guided_gene_ids_intersection
from models import ViT_PMA
from models.rna_predictor import RNAPredictionHead
from utils.metrics import compute_survival_metrics

BRANCHES = ["wsi", "rna"]
N_FOLDS = 5
SEED = 84
CKPT_PREFIX = "survival_tcga_uni_seed84_INT1500_SS_AUX_PMA_uni_INT1500_SS_AUX_NOCLINICAL_DISP_FOLD"


@torch.no_grad()
def _patient_forward(model, patient_slides, device):
    rna = patient_slides[0]["rna"].to(device, non_blocking=True)
    z_rna = model.encode_rna(rna)

    components_per_slide = []
    spatial_feats = []
    for slide in patient_slides:
        coords = slide["coords"].to(device, non_blocking=True)
        out = model(coords, features=slide["features"])
        components_per_slide.append(out["embed"])
        if "spatial_feat" in out:
            spatial_feats.append(out["spatial_feat"])
    patient_components = torch.stack(components_per_slide).mean(dim=0)
    spatial_feat = torch.stack(spatial_feats).mean(dim=0) if spatial_feats else None

    z_wsi, coattn_weights = model.component_coattn(patient_components, z_rna)

    return {
        "z_wsi": z_wsi.float().cpu(), "z_rna": z_rna.float().cpu(),
        "spatial_feat": spatial_feat.float().cpu() if spatial_feat is not None else None,
        "coattn_weights": coattn_weights.float().cpu(),
        "case_id": patient_slides[0]["case_id"],
        "time": float(patient_slides[0]["OS_time"].item()),
        "event": int(patient_slides[0]["OS_event"].item()),
    }


def _collect(model, dataset, device) -> list[dict]:
    return [_patient_forward(model, dataset[i], device) for i in range(len(dataset))]


@torch.no_grad()
def _fold_risks(model, records: list[dict], device) -> dict[str, np.ndarray]:
    parts = {
        "wsi": torch.stack([r["z_wsi"] for r in records]).to(device),
        "rna": torch.stack([r["z_rna"] for r in records]).to(device),
    }
    spatial_feat = (
        torch.stack([r["spatial_feat"] for r in records]).to(device)
        if records[0]["spatial_feat"] is not None else None
    )

    def _batch_risk(p: dict) -> np.ndarray:
        pieces = [p["wsi"], p["rna"]]
        if spatial_feat is not None:
            pieces.append(spatial_feat)
        combined = torch.cat(pieces, dim=-1)
        return model.risk_head(combined).view(-1).cpu().numpy()

    out = {"baseline": _batch_risk(parts)}
    for branch in BRANCHES:
        zp = dict(parts)
        zp[branch] = torch.zeros_like(parts[branch])
        out[f"zero_{branch}"] = _batch_risk(zp)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.model.use_attn_dispersion = True
    cfg.data.seed = SEED  # _kfold_case_split이 cfg.data.seed로 fold를 나눔 — 학습 때와 반드시 일치해야 함
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)
    ckpt_dir = Path(__file__).resolve().parent.parent / "models" / "checkpoint"

    ds_kwargs = dict(with_clinical=False, with_rna=True, rna_gene_ids=rna_gene_ids, feature_backbone="uni")

    external_ds = WSISurvivalDataset(cfg.data, dataset="cptac", split="all", **ds_kwargs)
    print(f"external(cptac 전체) 환자 수: {len(external_ds)}")

    pooled_internal = {"baseline": [], "zero_wsi": [], "zero_rna": []}
    pooled_internal_time, pooled_internal_event = [], []
    external_fold_metrics = {k: [] for k in ["baseline", "zero_wsi", "zero_rna"]}

    for fold in range(N_FOLDS):
        ckpt_path = ckpt_dir / f"{CKPT_PREFIX}{fold}OF{N_FOLDS}_best_pma.pt"
        print(f"\n=== fold {fold} 체크포인트: {ckpt_path.name} ===")

        model = ViT_PMA(cfg.model, age_mean=0.0, age_std=1.0, rna_input_dim=len(rna_gene_ids),
                         backbone="uni", use_clinical=False).to(device)
        model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, len(rna_gene_ids)).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        internal_ds = WSISurvivalDataset(cfg.data, dataset="tcga", split="test", fold=fold, n_folds=N_FOLDS, **ds_kwargs)
        internal_records = _collect(model, internal_ds, device)
        internal_risks = _fold_risks(model, internal_records, device)
        for k, v in internal_risks.items():
            pooled_internal[k].append(v)
        pooled_internal_time.append(np.array([r["time"] for r in internal_records]))
        pooled_internal_event.append(np.array([r["event"] for r in internal_records]))

        external_records = _collect(model, external_ds, device)
        external_risks = _fold_risks(model, external_records, device)
        ext_time = np.array([r["time"] for r in external_records])
        ext_event = np.array([r["event"] for r in external_records])
        for k, v in external_risks.items():
            external_fold_metrics[k].append(compute_survival_metrics(v, ext_time, ext_event))

    print(f"\n{'='*70}\nInternal (5-fold pooled, N={sum(len(t) for t in pooled_internal_time)})\n{'='*70}")
    time_all = np.concatenate(pooled_internal_time)
    event_all = np.concatenate(pooled_internal_event)
    baseline_c = None
    for cond in ["baseline", "zero_wsi", "zero_rna"]:
        risk_all = np.concatenate(pooled_internal[cond])
        m = compute_survival_metrics(risk_all, time_all, event_all)
        if cond == "baseline":
            baseline_c = m["c_index"]
        delta = "" if cond == "baseline" else f"  (baseline 대비 {m['c_index']-baseline_c:+.4f})"
        print(f"  [{cond:10s}] C={m['c_index']:.4f}  HR={m['hr']:.3f}  logrank_p={m['log_rank_p']:.4f}{delta}")

    print(f"\n{'='*70}\nExternal (5-fold 평균±표준편차, N=144 x 5)\n{'='*70}")
    base_c_mean = np.mean([m["c_index"] for m in external_fold_metrics["baseline"]])
    for cond in ["baseline", "zero_wsi", "zero_rna"]:
        cs = [m["c_index"] for m in external_fold_metrics[cond]]
        delta = "" if cond == "baseline" else f"  (baseline 대비 {np.mean(cs)-base_c_mean:+.4f})"
        print(f"  [{cond:10s}] C={np.mean(cs):.4f} +/- {np.std(cs):.4f}{delta}")


if __name__ == "__main__":
    main()
