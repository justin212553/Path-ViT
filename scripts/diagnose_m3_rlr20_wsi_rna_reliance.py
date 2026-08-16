"""
M3(=M4-NOVIT minus clinical, --rna-lr-mult 20.0, seed42)에서 WSI 브랜치와 RNA 브랜치가 최종
risk score에 얼마나 기여하는지 zero-ablation으로 직접 측정한다 — scripts/
diagnose_m2_branch_swap.py와 동일 방법론(해당 브랜치를 0으로 치환한 뒤 risk_head만
재실행, c-index 하락폭 + raw risk 변화폭으로 기여도 측정), M2(WSI vs Clinical) 대신
M3(WSI vs RNA)에 적용한 버전.

배경(2026-08-15): "RNA가 WSI를 항상 압도한다"는 기억은 확인해보니 이번 세션의 M3(NOVIT)+
RLR20이 아니라 훨씬 이전 세션의 PMA(UNI 백본, seed84) 진단이었다 — 지금 모델로 직접
측정한 적이 없어서 확인한다. M3는 combine_mode가 항상 concat이라(clinical이 없어
cox_add 분기가 자동으로 막힘, models/vit_m4.py::combine_with_clinical_rna) z_wsi/z_rna가
risk_head 입력에 그대로 concat돼 있어 branch-swap이 M2와 동일한 방식으로 바로 적용된다.

사용법: python -m scripts.diagnose_m3_rlr20_wsi_rna_reliance
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
from models import ViT_M4
from models.rna_predictor import RNAPredictionHead
from utils.metrics import compute_survival_metrics

FOLD, N_FOLDS = 1, 5
SEED = 42
CKPT_DIR = _ROOT / "models" / "checkpoint"
CKPT_GLOB = f"survival_tcga_uni2_seed{SEED}_*M4_uni2_INT1500_SS_AUX_NOCLINICAL_DISP_NOVIT_RLR20_FOLD{FOLD}OF{N_FOLDS}_best_clinical_rna.pt"


@torch.no_grad()
def _patient_forward(model, patient_slides, device):
    rna = patient_slides[0]["rna"].to(device, non_blocking=True)
    z_rna = model.encode_rna(rna)  # (D,)

    slide_embeds, slide_spatial = [], []
    for slide in patient_slides:
        coords = slide["coords"].to(device, non_blocking=True)
        out = model(coords, features=slide["features"], rna_context=z_rna)
        slide_embeds.append(out["embed"])
        if "spatial_feat" in out:
            slide_spatial.append(out["spatial_feat"])
    z_wsi = torch.stack(slide_embeds).mean(dim=0)  # (D,)
    spatial = torch.stack(slide_spatial).mean(dim=0) if slide_spatial else None

    return {
        "z_wsi": z_wsi.float().cpu(),
        "z_rna": z_rna.float().cpu(),
        "spatial": spatial.float().cpu() if spatial is not None else None,
        "time": float(patient_slides[0]["OS_time"].item()),
        "event": int(patient_slides[0]["OS_event"].item()),
    }


def _report(label: str, risk: np.ndarray, times: np.ndarray, events: np.ndarray):
    m = compute_survival_metrics(risk, times, events)
    print(f"  [{label:22s}] C={m['c_index']:.4f}  HR={m['hr']:.3f}  logrank_p={m['log_rank_p']:.4f}  n={len(risk)}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.model.use_attn_dispersion = True
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)

    ckpt_matches = list(CKPT_DIR.glob(CKPT_GLOB))
    if len(ckpt_matches) != 1:
        raise RuntimeError(f"checkpoint {len(ckpt_matches)}개 매칭(1개여야 함): {CKPT_GLOB}")
    model = ViT_M4(
        cfg.model, age_mean=0.0, age_std=1.0, rna_input_dim=len(rna_gene_ids),
        precomputed=True, backbone="uni2", use_attn_dispersion=True, combine_mode="concat",
        skip_patch_vit=True, use_clinical=False,
    ).to(device)
    model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, len(rna_gene_ids)).to(device)
    ckpt = torch.load(ckpt_matches[0], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"체크포인트: {ckpt_matches[0].name} (epoch={ckpt.get('epoch')} val_c={ckpt.get('val_c_index')})")

    ds_kwargs = dict(with_clinical=False, with_rna=True, rna_gene_ids=rna_gene_ids, feature_backbone="uni2")
    internal_ds = WSISurvivalDataset(cfg.data, dataset="tcga", split="test", fold=FOLD, n_folds=N_FOLDS, **ds_kwargs)
    external_ds = WSISurvivalDataset(cfg.data, dataset="cptac", split="all", **ds_kwargs)
    print(f"internal(fold{FOLD} held-out): {len(internal_ds)}명 | external(cptac): {len(external_ds)}명")

    for label, ds in [("internal(tcga fold1 held-out)", internal_ds), ("external(cptac)", external_ds)]:
        print(f"\n=== {label} ===")
        records = [_patient_forward(model, ds[i], device) for i in range(len(ds))]
        times = np.array([r["time"] for r in records])
        events = np.array([r["event"] for r in records])

        with torch.no_grad():
            z_wsi = torch.stack([r["z_wsi"] for r in records]).to(device)
            z_rna = torch.stack([r["z_rna"] for r in records]).to(device)
            has_spatial = records[0]["spatial"] is not None
            spatial = torch.stack([r["spatial"] for r in records]).to(device) if has_spatial else None

            def _risk(wsi, rna):
                fused = torch.cat([wsi, rna], dim=-1)
                if has_spatial:
                    fused = torch.cat([fused, spatial], dim=-1)
                return model.risk_head(fused).view(-1).cpu().numpy()

            baseline_risk = _risk(z_wsi, z_rna)
            baseline_c = compute_survival_metrics(baseline_risk, times, events)["c_index"]

            zero_wsi_risk = _risk(torch.zeros_like(z_wsi), z_rna)
            zero_wsi_c = compute_survival_metrics(zero_wsi_risk, times, events)["c_index"]

            zero_rna_risk = _risk(z_wsi, torch.zeros_like(z_rna))
            zero_rna_c = compute_survival_metrics(zero_rna_risk, times, events)["c_index"]

            wsi_std = float((baseline_risk - zero_wsi_risk).std())
            rna_std = float((baseline_risk - zero_rna_risk).std())

        print(f"  baseline c_index={baseline_c:.4f}")
        print(f"  [wsi 끄면] c={zero_wsi_c:.4f} (diff {zero_wsi_c-baseline_c:+.4f})  raw risk 변화폭(std)={wsi_std:.4f}")
        print(f"  [rna 끄면] c={zero_rna_c:.4f} (diff {zero_rna_c-baseline_c:+.4f})  raw risk 변화폭(std)={rna_std:.4f}")
        print(f"  RNA/WSI 기여 비율(std) = {rna_std/max(wsi_std,1e-8):.3f}  (1보다 크면 RNA가 더 많이 흔듦)")


if __name__ == "__main__":
    main()
