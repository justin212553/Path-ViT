"""
M2(WSI+Clinical, concat, seed42 fold0)의 WSI 브랜치/clinical 브랜치를 각각 독립 모델
(M1=WSI 단독, M5_STG_R=Clinical 단독)의 risk_head에 "이식"해서 채점해본다 — M2/M4가 M3를
못 넘는 게 "브랜치 표현 자체가 공동학습으로 나빠진 것"인지 "브랜치는 멀쩡한데 결합
risk_head/fusion이 문제인지" 구분하는 게 목적.

측정 4가지 (내부 fold0 held-out + external cptac 둘 다):
  (A) M1 네이티브: M1의 z_wsi -> M1의 risk_head (기준선)
  (B) M2->M1 이식: M2의 z_wsi(공동학습됨) -> M1의 risk_head
      (A)와 큰 차이 없으면 WSI 표현은 안 상했다는 뜻, 확 떨어지면 공동학습이 WSI 표현
      자체를 나쁘게 만들었다는 뜻.
  (C) M5 네이티브: M5의 z_clinical -> M5의 risk_head (기준선)
  (D) M2->M5 이식: M2의 z_clinical(공동학습됨) -> M5의 risk_head
      (C)와 비교 방식은 (A)-(B)와 동일.

사용법: python -m scripts.diagnose_m2_branch_swap
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
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS
from models import ViT_M1, ViT_M2, ClinicalOnly
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_csv
from train import _stage_ord_from_patient, _margin_ord_from_patient
from utils.metrics import compute_survival_metrics

FOLD, N_FOLDS = 1, 5
SEED = 42
CKPT_DIR = _ROOT / "models" / "checkpoint"

M1_CKPT = CKPT_DIR / f"survival_tcga_uni2_seed{SEED}_M1_uni2_DISP_NOVIT_FOLD{FOLD}OF{N_FOLDS}_best.pt"
M2_CKPT_GLOB = f"survival_tcga_uni2_seed{SEED}_*M2_uni2_STG_R_DISP_NOVIT_FOLD{FOLD}OF{N_FOLDS}_best_clinical.pt"
M5_CKPT = CKPT_DIR / f"survival_tcga_best_m5_stg_r_fold{FOLD}of{N_FOLDS}_seed{SEED}_light.pt"


def _identity_collate(batch):
    return batch[0]


@torch.no_grad()
def _m1_wsi_embed(model, patient_slides, device):
    slide_embeds, slide_spatial = [], []
    for slide in patient_slides:
        coords = slide["coords"].to(device)
        out = model(coords, features=slide["features"])
        slide_embeds.append(out["embed"])
        if "spatial_feat" in out:
            slide_spatial.append(out["spatial_feat"])
    z_wsi = torch.stack(slide_embeds).mean(dim=0)
    spatial = torch.stack(slide_spatial).mean(dim=0) if slide_spatial else None
    return z_wsi.float().cpu(), (spatial.float().cpu() if spatial is not None else None)


@torch.no_grad()
def _m2_clinical_embed(model, patient_slides, device):
    age_years = patient_slides[0]["age_years"].to(device)
    sex_idx = patient_slides[0]["sex_idx"].to(device)
    stage_ord = _stage_ord_from_patient(patient_slides, device)
    margin_ord = _margin_ord_from_patient(patient_slides, device)
    clinical_kwargs = {}
    if stage_ord is not None:
        clinical_kwargs["stage_ord"] = {k: v.unsqueeze(0) for k, v in stage_ord.items()}
    if margin_ord is not None:
        clinical_kwargs["margin_ord"] = margin_ord.unsqueeze(0)
    z_clinical = model.clinical_encoder(
        age_years.unsqueeze(0), sex_idx.unsqueeze(0), **clinical_kwargs
    ).squeeze(0)
    return z_clinical.float().cpu()


@torch.no_grad()
def _collect(ds, device, m1_model, m2_model):
    records = []
    for i in range(len(ds)):
        patient_slides = ds[i]
        z_wsi_m1, spatial_m1 = _m1_wsi_embed(m1_model, patient_slides, device)
        z_wsi_m2, spatial_m2 = _m1_wsi_embed(m2_model, patient_slides, device)  # M2도 forward 시그니처 동일(ViT_M1 상속)
        z_clinical_m2 = _m2_clinical_embed(m2_model, patient_slides, device)
        records.append({
            "case_id": patient_slides[0]["case_id"],
            "z_wsi_m1": z_wsi_m1, "spatial_m1": spatial_m1,
            "z_wsi_m2": z_wsi_m2, "spatial_m2": spatial_m2,
            "z_clinical_m2": z_clinical_m2,
            "age_years": patient_slides[0]["age_years"].cpu(),
            "sex_idx": patient_slides[0]["sex_idx"].cpu(),
            "stage_ord": _stage_ord_from_patient(patient_slides, torch.device("cpu")),
            "margin_ord": _margin_ord_from_patient(patient_slides, torch.device("cpu")),
            "time": float(patient_slides[0]["OS_time"].item()),
            "event": int(patient_slides[0]["OS_event"].item()),
        })
    return records


def _report(label: str, risk: np.ndarray, times: np.ndarray, events: np.ndarray):
    m = compute_survival_metrics(risk, times, events)
    print(f"  [{label:22s}] C={m['c_index']:.4f}  HR={m['hr']:.3f}  logrank_p={m['log_rank_p']:.4f}  n={len(risk)}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.model.use_attn_dispersion = True
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS["tcga"])
    stage_stats = stage_stats_from_csv(CLINICAL_PATHS["tcga"])

    m1_model = ViT_M1(cfg.model, precomputed=True, backbone="uni2", use_attn_dispersion=True,
                       skip_patch_vit=True).to(device)
    m1_ckpt = torch.load(M1_CKPT, map_location=device, weights_only=False)
    m1_model.load_state_dict(m1_ckpt["model_state_dict"])
    m1_model.eval()

    m2_matches = list(CKPT_DIR.glob(M2_CKPT_GLOB))
    if len(m2_matches) != 1:
        raise RuntimeError(f"M2 checkpoint {len(m2_matches)}개 매칭(1개여야 함): {M2_CKPT_GLOB}")
    m2_model = ViT_M2(cfg.model, age_mean=age_mean, age_std=age_std, precomputed=True, backbone="uni2",
                       use_staging=True, stage_stats=stage_stats, use_margin=True, margin_stats=margin_stats,
                       use_attn_dispersion=True, combine_mode="concat", skip_patch_vit=True).to(device)
    m2_ckpt = torch.load(m2_matches[0], map_location=device, weights_only=False)
    m2_model.load_state_dict(m2_ckpt["model_state_dict"])
    m2_model.eval()
    print(f"M2 체크포인트: {m2_matches[0].name} (epoch={m2_ckpt.get('epoch')} val_c={m2_ckpt.get('val_c_index')})")

    m5_model = ClinicalOnly(cfg.model, age_mean=age_mean, age_std=age_std, use_staging=True,
                             stage_stats=stage_stats, use_margin=True, margin_stats=margin_stats).to(device)
    m5_ckpt = torch.load(M5_CKPT, map_location=device, weights_only=False)
    m5_model.load_state_dict(m5_ckpt["model_state_dict"])
    m5_model.eval()

    ds_kwargs = dict(with_clinical=True, with_margin=True, with_staging=True, feature_backbone="uni2")
    internal_ds = WSISurvivalDataset(cfg.data, dataset="tcga", split="test", fold=FOLD, n_folds=N_FOLDS, **ds_kwargs)
    external_ds = WSISurvivalDataset(cfg.data, dataset="cptac", split="all", **ds_kwargs)
    print(f"internal(fold{FOLD} held-out): {len(internal_ds)}명 | external(cptac): {len(external_ds)}명")

    for label, ds in [("internal(tcga fold0 held-out)", internal_ds), ("external(cptac)", external_ds)]:
        print(f"\n=== {label} ===")
        records = _collect(ds, device, m1_model, m2_model)
        times = np.array([r["time"] for r in records])
        events = np.array([r["event"] for r in records])

        with torch.no_grad():
            # (A) M1 네이티브
            z_wsi_m1 = torch.stack([r["z_wsi_m1"] for r in records]).to(device)
            sp_m1 = torch.stack([r["spatial_m1"] for r in records]).to(device) if records[0]["spatial_m1"] is not None else None
            wsi_in_a = torch.cat([z_wsi_m1, sp_m1], dim=-1) if sp_m1 is not None else z_wsi_m1
            risk_a = m1_model.risk_head(wsi_in_a).view(-1).cpu().numpy()

            # (B) M2 WSI 브랜치 -> M1 risk_head
            z_wsi_m2 = torch.stack([r["z_wsi_m2"] for r in records]).to(device)
            sp_m2 = torch.stack([r["spatial_m2"] for r in records]).to(device) if records[0]["spatial_m2"] is not None else None
            wsi_in_b = torch.cat([z_wsi_m2, sp_m2], dim=-1) if sp_m2 is not None else z_wsi_m2
            risk_b = m1_model.risk_head(wsi_in_b).view(-1).cpu().numpy()

            # (C) M5 네이티브
            age_years = torch.stack([r["age_years"] for r in records]).to(device)
            sex_idx = torch.stack([r["sex_idx"] for r in records]).to(device)
            risks_c = []
            for r in records:
                stage_ord = {k: v.to(device) for k, v in r["stage_ord"].items()} if r["stage_ord"] else None
                margin_ord = r["margin_ord"].to(device) if r["margin_ord"] is not None else None
                risk = m5_model(r["age_years"].to(device), r["sex_idx"].to(device),
                                 stage_ord=stage_ord, margin_ord=margin_ord)
                risks_c.append(risk.item())
            risk_c = np.array(risks_c)

            # (D) M2 clinical 브랜치 -> M5 risk_head
            z_clinical_m2 = torch.stack([r["z_clinical_m2"] for r in records]).to(device)
            risk_d = m5_model.risk_head(z_clinical_m2).view(-1).cpu().numpy()

        _report("(A) M1 네이티브(WSI)", risk_a, times, events)
        _report("(B) M2WSI->M1head", risk_b, times, events)
        _report("(C) M5 네이티브(Clin)", risk_c, times, events)
        _report("(D) M2Clin->M5head", risk_d, times, events)


if __name__ == "__main__":
    main()
