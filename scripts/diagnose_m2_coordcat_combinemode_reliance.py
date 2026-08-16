"""
M2(WSI+Clinical, coord-embed-concat 적용, seed42 fold0)의 두 combine_mode(cox_add vs concat)
에서 WSI 브랜치와 clinical 브랜치가 최종 risk score에 얼마나 기여하는지 zero-ablation으로
비교한다 — scripts/diagnose_pma_agesex_reliance.py와 동일 방법(해당 브랜치를 0으로 치환한 뒤
risk_head(+cox_add면 clinical_linear)만 재실행, c-index 하락폭으로 기여도 측정).

배경(2026-08-15): coord-embed-concat을 M2에 적용했더니 cox_add 모드에서 internal/external
둘 다 하락(0.5529->0.5155, 0.5291->0.5111)했는데, M2를 concat 모드로 바꾸면(=PMA에서 clinical과
WSI+RNA가 cox_add로 상호작용할 때 나쁘게 작용했던 것과 같은 클래스의 문제일 수 있다는 가설)
fold0 파일럿에서 방향이 나아지는 것으로 보였다(test_c_index 0.3860->0.4702). 이 스크립트로
"clinical/WSI 비율이 두 모드에서 실제로 어떻게 다른가"를 직접 측정해 그 가설을 뒷받침하는지
확인한다.

사용법: python -m scripts.diagnose_m2_coordcat_combinemode_reliance
"""
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS
from models import ViT_M2
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_csv
from train import _stage_ord_from_patient, _margin_ord_from_patient
from utils.metrics import compute_survival_metrics

FOLD, N_FOLDS = 0, 5
SEED = 42
CKPT_DIR = _ROOT / "models" / "checkpoint"
CKPT_GLOBS = {
    "cox_add": f"survival_tcga_uni2_seed{SEED}_*M2_uni2_STG_R_DISP_COX_ADD_NOVIT_COORD_CAT_FOLD{FOLD}OF{N_FOLDS}_best_clinical.pt",
    "concat":  f"survival_tcga_uni2_seed{SEED}_*M2_uni2_STG_R_DISP_NOVIT_COORD_CAT_FOLD{FOLD}OF{N_FOLDS}_best_clinical.pt",
}


def _identity_collate(batch):
    return batch[0]


@torch.no_grad()
def _patient_forward(model, patient_slides, device, combine_mode: str):
    """(z_wsi+spatial_feat concat, clinical_raw 또는 z_clinical, time, event)를 patient 단위로 뽑는다."""
    slide_embeds, slide_spatial_feats = [], []
    for slide in patient_slides:
        coords = slide["coords"].to(device, non_blocking=True)
        out = model(coords, features=slide["features"])
        slide_embeds.append(out["embed"])
        if "spatial_feat" in out:
            slide_spatial_feats.append(out["spatial_feat"])
    patient_embed = torch.stack(slide_embeds).mean(dim=0)  # (D,)
    spatial_feat = torch.stack(slide_spatial_feats).mean(dim=0) if slide_spatial_feats else None

    age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
    sex_idx = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
    stage_ord = _stage_ord_from_patient(patient_slides, device)
    margin_ord = _margin_ord_from_patient(patient_slides, device)

    if combine_mode == "cox_add":
        clinical_raw = model._clinical_raw(age_years, sex_idx, margin_ord, stage_ord=stage_ord).squeeze(0)  # (raw_dim,)
        z_clinical = clinical_raw  # cox_add는 clinical을 임베딩 안 하고 raw feature를 그대로 씀
    else:
        clinical_kwargs = {}
        if stage_ord is not None:
            clinical_kwargs["stage_ord"] = {k: v.unsqueeze(0) for k, v in stage_ord.items()}
        if margin_ord is not None:
            clinical_kwargs["margin_ord"] = margin_ord.unsqueeze(0)
        z_clinical = model.clinical_encoder(
            age_years.unsqueeze(0), sex_idx.unsqueeze(0), **clinical_kwargs
        ).squeeze(0)  # (D,)

    return {
        "z_wsi": patient_embed.float().cpu(),
        "spatial_feat": spatial_feat.float().cpu() if spatial_feat is not None else None,
        "z_clinical": z_clinical.float().cpu(),
        "time": float(patient_slides[0]["OS_time"].item()),
        "event": int(patient_slides[0]["OS_event"].item()),
    }


def _collect(model, dataset, device, combine_mode: str) -> list[dict]:
    return [_patient_forward(model, dataset[i], device, combine_mode) for i in range(len(dataset))]


@torch.no_grad()
def _ablation_report(model, records: list[dict], device, combine_mode: str) -> dict:
    z_wsi = torch.stack([r["z_wsi"] for r in records]).to(device)          # (N, D)
    z_clinical = torch.stack([r["z_clinical"] for r in records]).to(device)  # (N, D) 또는 (N, raw_dim)
    has_spatial = records[0]["spatial_feat"] is not None
    spatial = torch.stack([r["spatial_feat"] for r in records]).to(device) if has_spatial else None
    times = np.array([r["time"] for r in records])
    events = np.array([r["event"] for r in records])

    def _risk(wsi, clinical):
        if combine_mode == "cox_add":
            wsi_in = torch.cat([wsi, spatial], dim=-1) if has_spatial else wsi
            risk = model.risk_head(wsi_in).view(-1)
            risk = risk + model.clinical_linear(clinical).view(-1)
        else:  # concat
            fused = torch.cat([wsi, clinical], dim=-1)
            if has_spatial:
                fused = torch.cat([fused, spatial], dim=-1)
            risk = model.risk_head(fused).view(-1)
        return risk.cpu().numpy()

    baseline_risk = _risk(z_wsi, z_clinical)
    baseline_metrics = compute_survival_metrics(baseline_risk, times, events)

    zero_wsi_risk = _risk(torch.zeros_like(z_wsi), z_clinical)
    zero_wsi_metrics = compute_survival_metrics(zero_wsi_risk, times, events)

    zero_clinical_risk = _risk(z_wsi, torch.zeros_like(z_clinical))
    zero_clinical_metrics = compute_survival_metrics(zero_clinical_risk, times, events)

    # risk score 자체가 baseline 대비 얼마나 움직였는지(방향 무관, 절대적 기여 크기) — c-index만으론
    # "둘 다 거의 안 흔들리는데 우연히 순위만 비슷"한 경우를 못 잡아내므로 raw 변화폭도 같이 본다.
    wsi_contrib_std = float((baseline_risk - zero_wsi_risk).std())      # WSI를 껐을 때 risk가 흔들린 정도
    clinical_contrib_std = float((baseline_risk - zero_clinical_risk).std())

    return {
        "n": len(records),
        "baseline_c": baseline_metrics["c_index"],
        "wsi": {"zero_c": zero_wsi_metrics["c_index"], "contrib_std": wsi_contrib_std},
        "clinical": {"zero_c": zero_clinical_metrics["c_index"], "contrib_std": clinical_contrib_std},
    }


def _print_report(label: str, rep: dict):
    print(f"\n--- {label} (n={rep['n']}) ---")
    print(f"  baseline c_index={rep['baseline_c']:.4f}")
    for branch in ("wsi", "clinical"):
        br = rep[branch]
        print(f"  [{branch:9s}] zero-ablation c={br['zero_c']:.4f}  (baseline 대비 {br['zero_c']-rep['baseline_c']:+.4f})"
              f"   |  raw risk 변화폭(std)={br['contrib_std']:.4f}")
    ratio = rep["wsi"]["contrib_std"] / max(rep["clinical"]["contrib_std"], 1e-8)
    print(f"  WSI/clinical raw 기여 비율(std) = {ratio:.3f}  (1보다 크면 WSI가 더 많이 흔듦)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.model.use_attn_dispersion = True
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS["tcga"])
    stage_stats = stage_stats_from_csv(CLINICAL_PATHS["tcga"])

    internal_ds = WSISurvivalDataset(
        cfg.data, dataset="tcga", split="test", fold=FOLD, n_folds=N_FOLDS,
        with_clinical=True, with_margin=True, with_staging=True, feature_backbone="uni2",
    )
    external_ds = WSISurvivalDataset(
        cfg.data, dataset="cptac", split="all",
        with_clinical=True, with_margin=True, with_staging=True, feature_backbone="uni2",
    )
    print(f"internal(tcga fold{FOLD} held-out test) 환자 수: {len(internal_ds)}")
    print(f"external(cptac 전체) 환자 수: {len(external_ds)}")

    for combine_mode, glob_pat in CKPT_GLOBS.items():
        ckpt_matches = list(CKPT_DIR.glob(glob_pat))
        if len(ckpt_matches) != 1:
            print(f"\n[SKIP] combine_mode={combine_mode}: checkpoint {len(ckpt_matches)}개 매칭 (1개여야 함): {glob_pat}")
            continue
        ckpt_path = ckpt_matches[0]
        print(f"\n===== combine_mode={combine_mode} — {ckpt_path.name} =====")

        model = ViT_M2(
            cfg.model, age_mean=age_mean, age_std=age_std, precomputed=True, backbone="uni2",
            use_staging=True, stage_stats=stage_stats, use_margin=True, margin_stats=margin_stats,
            use_attn_dispersion=True, combine_mode=combine_mode, skip_patch_vit=True,
            use_coord_embed=True, coord_embed_concat=True,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"체크포인트: epoch={ckpt.get('epoch')} val_c_index={ckpt.get('val_c_index')}")

        internal_records = _collect(model, internal_ds, device, combine_mode)
        external_records = _collect(model, external_ds, device, combine_mode)

        internal_rep = _ablation_report(model, internal_records, device, combine_mode)
        external_rep = _ablation_report(model, external_records, device, combine_mode)

        _print_report(f"internal(tcga fold{FOLD} held-out)", internal_rep)
        _print_report("external(cptac)", external_rep)


if __name__ == "__main__":
    main()
