"""
PMA(ViT_PMA)의 MultiComponentPooling 4개 관점(mean/std/attn/top-k) 중 어느 것에 실제로
의존하는지 확인하는 branch-ablation 진단. 이미 학습된 체크포인트를 그대로 두고, 추론
시점에 4개 성분 중 하나씩만 0벡터로 바꿔 co-attention(RNA query)에 넣은 뒤 c-index 변화를
본다 — scripts/bootstrap_pma_internal_external_gap.py의 external forward 로직을 재사용하되,
"어느 component를 지울지"만 바꿔가며 5가지(baseline + 4개 ablation)를 비교한다.

internal(그 checkpoint를 만든 seed의 단일 6:2:2 test split, 표본 작아 노이즈 큼)과
external(cptac 전체, N=144, 상대적으로 안정적) 둘 다 계산한다.

사용법: python -m scripts.diagnose_pma_component_reliance --seed 126 --ckpt <경로>
"""
import argparse
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
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_csv, STAGE_FIELDS
from models.rna_predictor import RNAPredictionHead
from utils.metrics import compute_survival_metrics

COMPONENT_NAMES = ["mean", "std", "attn", "top-k"]  # models/multi_component_pooling.py 순서와 동일


def _identity_collate(batch):
    return batch[0]


def _build_model(device, rna_input_dim, age_mean, age_std, margin_stats, stage_stats, ckpt_path):
    cfg = Config()
    cfg.model.use_attn_dispersion = True
    model = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                     backbone="uni2", combine_mode="cox_add", use_margin=True, margin_stats=margin_stats,
                     use_age_sex=True, use_staging=True, stage_stats=stage_stats).to(device)
    model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, rna_input_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"체크포인트 로드 완료: {ckpt_path} (epoch={ckpt.get('epoch')}, val_c_index={ckpt.get('val_c_index')})")
    return model


@torch.no_grad()
def _collect_patient_data(model, loader, device):
    """환자별 (4,D) component, z_rna, clinical raw 입력, 라벨을 한 번만 forward해서 모아둔다 —
    ablation 5가지를 매번 다시 forward하지 않고 이 캐시로 risk만 재계산해 속도를 낸다."""
    patients = []
    for patient_slides in loader:
        p = patient_slides[0]
        rna = p["rna"].to(device)
        z_rna = model.encode_rna(rna)
        comps, spatial_feats = [], []
        for slide in patient_slides:
            out = model(slide["coords"].to(device), features=slide["features"])
            comps.append(out["embed"])
            if "spatial_feat" in out:
                spatial_feats.append(out["spatial_feat"])
        patient_embed = torch.stack(comps).mean(dim=0)  # (4, D)
        spatial_feat = torch.stack(spatial_feats).mean(dim=0) if spatial_feats else None

        age_years = p["age_years"].to(device)
        sex_idx = p["sex_idx"].to(device)
        margin_ord = p["margin_ord"].to(device)
        stage_ord = {f: p[f].to(device) for f in STAGE_FIELDS}
        clin_raw = model._clinical_raw(age_years, sex_idx, margin_ord, stage_ord=stage_ord)

        patients.append({
            "case_id": p["case_id"], "time": float(p["OS_time"].item()), "event": int(p["OS_event"].item()),
            "patient_embed": patient_embed, "z_rna": z_rna, "spatial_feat": spatial_feat, "clin_raw": clin_raw,
        })
    return patients


@torch.no_grad()
def _risk_with_ablation(model, patients, ablate_idx: int | None):
    """ablate_idx=None이면 baseline(4개 다 사용), 0~3이면 그 성분을 0으로 지운다."""
    risks, times, events = [], [], []
    for pd in patients:
        embed = pd["patient_embed"].clone()
        if ablate_idx is not None:
            embed[ablate_idx] = 0.0
        z_wsi, _ = model.component_coattn(embed, pd["z_rna"])
        fused = torch.cat([z_wsi, pd["z_rna"]], dim=-1)
        if pd["spatial_feat"] is not None:
            fused = torch.cat([fused, pd["spatial_feat"]], dim=-1)
        risk = model.risk_head(fused.unsqueeze(0)).view(1)
        risk = risk + model.clinical_linear(pd["clin_raw"]).view(1)
        risks.append(risk.item())
        times.append(pd["time"])
        events.append(pd["event"])
    return compute_survival_metrics(np.array(risks), np.array(times), np.array(events))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.data.seed = args.seed  # _kfold_case_split/단일 split이 이 seed로 갈리므로 반드시 맞춰야 함

    rna_gene_ids = literature_guided_gene_ids_intersection(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS["tcga"])
    stage_stats = stage_stats_from_csv(CLINICAL_PATHS["tcga"])

    model = _build_model(device, len(rna_gene_ids), age_mean, age_std, margin_stats, stage_stats, args.ckpt)

    ds_kwargs = dict(with_clinical=True, with_margin=True, with_staging=True, with_rna=True,
                      rna_gene_ids=rna_gene_ids, feature_backbone="uni2")

    print("\n=== Internal (그 seed의 held-out test split) ===")
    internal_ds = WSISurvivalDataset(cfg.data, dataset="tcga", split="test", **ds_kwargs)
    internal_loader = DataLoader(internal_ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    internal_patients = _collect_patient_data(model, internal_loader, device)
    print(f"  N={len(internal_patients)}")
    _print_ablation_table(model, internal_patients)

    print("\n=== External (cptac 전체 코호트) ===")
    external_ds = WSISurvivalDataset(cfg.data, dataset="cptac", split="all", **ds_kwargs)
    external_loader = DataLoader(external_ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    external_patients = _collect_patient_data(model, external_loader, device)
    print(f"  N={len(external_patients)}")
    _print_ablation_table(model, external_patients)


def _print_ablation_table(model, patients):
    baseline = _risk_with_ablation(model, patients, None)
    print(f"  baseline(4개 다 사용): c_index={baseline['c_index']:.4f}")
    for i, name in enumerate(COMPONENT_NAMES):
        m = _risk_with_ablation(model, patients, i)
        delta = m["c_index"] - baseline["c_index"]
        print(f"  {name} 제거: c_index={m['c_index']:.4f}  (delta={delta:+.4f})")


if __name__ == "__main__":
    main()
