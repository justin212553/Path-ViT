"""
PMA_uni_INT1500_SS_AUX_R_DISP(seed84, 5-fold, UNI 백본) 체크포인트로 branch-ablation을 돌려,
UNI 백본에서 clinical을 추가했을 때(PMA_R) external이 clinical 없는 M3보다 오히려 크게
떨어지는 이유를 진단한다(scripts/diagnose_wsi_reliance.py와 동일한 방법론, fold 체크포인트
5개 버전으로 재구현 — 원본은 단일-split 체크포인트 전용이라 그대로 재사용이 안 됨).

internal은 5개 fold의 ablation 결과를 pool_kfold_preds.py와 동일하게 이어붙여(각 fold test
환자를 그 fold의 체크포인트로 한 번씩 평가) 코호트 전체 크기로 계산하고, external은 5개
체크포인트가 전부 같은 144명을 보므로 평균±표준편차로 요약한다.

사용법: python -m scripts.diagnose_pma_uni_r_reliance
"""
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids_intersection
from models import ViT_PMA
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv
from models.rna_predictor import RNAPredictionHead
from utils.metrics import compute_survival_metrics

COMPONENT_NAMES = ["mean", "std", "attn", "topk"]
BRANCHES = ["wsi", "clinical", "rna"]
N_FOLDS = 5
SEED = 84
CKPT_TAG = "pma_uni_int1500_ss_aux_r_disp"


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

    age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
    sex_idx = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
    margin_ord = patient_slides[0]["margin_ord"].to(device, non_blocking=True)
    z_clinical = model.clinical_encoder(
        age_years.unsqueeze(0), sex_idx.unsqueeze(0), margin_ord=margin_ord.unsqueeze(0)
    ).squeeze(0)

    z_wsi, coattn_weights = model.component_coattn(patient_components, z_rna)

    return {
        "z_wsi": z_wsi.float().cpu(), "z_clinical": z_clinical.float().cpu(), "z_rna": z_rna.float().cpu(),
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
    """이 fold 체크포인트로 baseline + 브랜치별 zero-ablation risk를 계산한다 (재학습 없이 risk_head만 재실행)."""
    parts = {
        "wsi": torch.stack([r["z_wsi"] for r in records]).to(device),
        "clinical": torch.stack([r["z_clinical"] for r in records]).to(device),
        "rna": torch.stack([r["z_rna"] for r in records]).to(device),
    }
    spatial_feat = (
        torch.stack([r["spatial_feat"] for r in records]).to(device)
        if records[0]["spatial_feat"] is not None else None
    )

    def _batch_risk(p: dict) -> np.ndarray:
        pieces = [p["wsi"], p["clinical"], p["rna"]]
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
    cfg.model.use_attn_dispersion = True  # 학습 시 --attn-dispersion(DISP) 사용 — dispersion_scale/risk_head 차원 맞추기 필수
    cfg.data.seed = SEED  # _kfold_case_split(data/dataset.py)이 cfg.data.seed로 fold를 나눔 — 이걸
    # 안 맞추면 학습 때와 다른 fold 분할이 나와(환자 구성 자체가 달라짐) ablation이 완전히 다른
    # 코호트를 보게 된다(실제로 이 버그로 처음엔 internal C=0.73이 나와 학습 때 0.61과 안 맞았음).
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS["tcga"])
    ckpt_dir = Path(__file__).resolve().parent.parent / "models" / "checkpoint"

    ds_kwargs = dict(with_clinical=True, with_margin=True, with_rna=True, rna_gene_ids=rna_gene_ids,
                      feature_backbone="uni")

    external_ds = WSISurvivalDataset(cfg.data, dataset="cptac", split="all", **ds_kwargs)
    print(f"external(cptac 전체) 환자 수: {len(external_ds)}")

    pooled_internal = {"baseline": [], "zero_wsi": [], "zero_clinical": [], "zero_rna": []}
    pooled_internal_time, pooled_internal_event = [], []
    external_fold_metrics = {k: [] for k in ["baseline", "zero_wsi", "zero_clinical", "zero_rna"]}

    for fold in range(N_FOLDS):
        ckpt_path = ckpt_dir / (
            f"survival_tcga_uni_seed{SEED}_INT1500_SS_AUX_R_PMA_uni_INT1500_SS_AUX_R_DISP_"
            f"FOLD{fold}OF{N_FOLDS}_best_pma.pt"
        )
        print(f"\n=== fold {fold} 체크포인트: {ckpt_path.name} ===")

        model = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(rna_gene_ids),
                         backbone="uni", use_margin=True, margin_stats=margin_stats, use_age_sex=True).to(device)
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
    for cond in ["baseline", "zero_wsi", "zero_clinical", "zero_rna"]:
        risk_all = np.concatenate(pooled_internal[cond])
        m = compute_survival_metrics(risk_all, time_all, event_all)
        if cond == "baseline":
            baseline_c = m["c_index"]
        delta = "" if cond == "baseline" else f"  (baseline 대비 {m['c_index']-baseline_c:+.4f})"
        print(f"  [{cond:14s}] C={m['c_index']:.4f}  HR={m['hr']:.3f}  logrank_p={m['log_rank_p']:.4f}{delta}")

    print(f"\n{'='*70}\nExternal (5-fold 평균±표준편차, N=144 x 5)\n{'='*70}")
    base_c_mean = np.mean([m["c_index"] for m in external_fold_metrics["baseline"]])
    for cond in ["baseline", "zero_wsi", "zero_clinical", "zero_rna"]:
        cs = [m["c_index"] for m in external_fold_metrics[cond]]
        delta = "" if cond == "baseline" else f"  (baseline 대비 {np.mean(cs)-base_c_mean:+.4f})"
        print(f"  [{cond:14s}] C={np.mean(cs):.4f} +/- {np.std(cs):.4f}{delta}")


if __name__ == "__main__":
    main()
