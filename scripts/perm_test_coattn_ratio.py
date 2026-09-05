"""
사용자 질문(2026-09-05): "변화량이 아니라 객관적 수치로 판단을 내린다면...? 가장 높은 점수는
충분히 유의한가?" — viz_pma_coattn_tsne.py가 찍는 "중심거리 비율"(_ratio)은 그냥 기술
통계량이라 0.3~0.4가 우연 수준인지 판단할 기준이 없다. 여기서는 OS_event/risk_tertile 라벨을
무작위로 섞어(permutation) 같은 비율을 반복 계산해서 관측값의 null 분포 대비 p-value를 낸다.

viz_pma_coattn_tsne.py와 동일한 hook/추출 로직을 재사용하되, t-SNE/플롯은 생략하고
BEFORE/AFTER × event/risk 4개 비율 전부에 permutation p-value를 붙인다.

사용법:
    python -m scripts.perm_test_coattn_ratio --ckpt <path> --backbone resnet50 --n-perm 5000
"""
import argparse
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
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, pdac_consistency_gene_ids
from models import ViT_PMA
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df
from train import _patient_risk, _identity_collate, _make_amp_ctx


def _ratio(X, mask_low, mask_high):
    low, high = X[mask_low], X[mask_high]
    if mask_low.sum() < 2 or mask_high.sum() < 2:
        return float("nan")
    d = np.linalg.norm(low.mean(0) - high.mean(0))
    s = np.linalg.norm(X - X.mean(0), axis=1).mean()
    return d / s


def _perm_test(X, mask_low, mask_high, n_perm, rng):
    observed = _ratio(X, mask_low, mask_high)
    n_low, n_high = mask_low.sum(), mask_high.sum()
    idx_pool = np.where(mask_low | mask_high)[0]
    n_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(idx_pool)
        plow = np.zeros(len(X), dtype=bool)
        phigh = np.zeros(len(X), dtype=bool)
        plow[perm[:n_low]] = True
        phigh[perm[n_low:n_low + n_high]] = True
        r = _ratio(X, plow, phigh)
        if r >= observed:
            n_ge += 1
    p = (n_ge + 1) / (n_perm + 1)
    return observed, p


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="uni2native")
    parser.add_argument("--train-dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=84)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx()
    external_dataset = {"tcga": "cptac", "cptac": "tcga"}[args.train_dataset]

    cfg = Config()
    rna_gene_ids = pdac_consistency_gene_ids(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.train_dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.train_dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.train_dataset]))

    ds_kwargs = dict(
        with_clinical=True, with_staging=True, with_margin=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone=args.backbone,
    )
    external_ds = WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", **ds_kwargs)
    loader = DataLoader(external_ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)

    rna_input_dim = len(rna_gene_ids) + 8
    model = ViT_PMA(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=cfg.data.precomputed, backbone=args.backbone, combine_mode="cox_add",
        use_staging=True, stage_stats=stage_stats, use_margin=True, margin_stats=margin_stats,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    captured = {}

    def _hook(module, inputs, output):
        patient_embed = inputs[0]
        z_wsi, _ = output
        captured["before"] = patient_embed.mean(dim=0).detach()
        captured["after"] = z_wsi.detach()

    handle = model.component_coattn.register_forward_hook(_hook)

    chunk_size = cfg.train.cnn_chunk_size
    rows, before_vecs, after_vecs = [], [], []
    with torch.no_grad():
        for patient_slides in loader:
            if not patient_slides:
                continue
            risk, _, _ = _patient_risk(model, patient_slides, device, amp_ctx, None, chunk_size)
            rows.append({
                "OS_event": int(patient_slides[0]["OS_event"].item()),
                "risk": risk.float().item(),
            })
            before_vecs.append(captured["before"].float().cpu().numpy())
            after_vecs.append(captured["after"].float().cpu().numpy())
    handle.remove()

    df = pd.DataFrame(rows)
    Xb, Xa = np.stack(before_vecs), np.stack(after_vecs)
    edges = np.quantile(df["risk"], [1 / 3, 2 / 3])
    df["risk_tertile"] = np.digitize(df["risk"], edges)

    ev_low, ev_high = (df["OS_event"] == 0).to_numpy(), (df["OS_event"] == 1).to_numpy()
    risk_low, risk_high = (df["risk_tertile"] == 0).to_numpy(), (df["risk_tertile"] == 2).to_numpy()
    print(f"ckpt={Path(args.ckpt).name}")
    print(f"N={len(df)} (event={ev_high.sum()}, censored={ev_low.sum()}, risk_low={risk_low.sum()}, risk_high={risk_high.sum()})")

    rng = np.random.default_rng(args.seed)
    print(f"\n=== permutation test (n_perm={args.n_perm}), observed ratio + p-value ===")
    for name, X in (("BEFORE", Xb), ("AFTER", Xa)):
        r_ev, p_ev = _perm_test(X, ev_low, ev_high, args.n_perm, rng)
        r_risk, p_risk = _perm_test(X, risk_low, risk_high, args.n_perm, rng)
        print(f"  {name:8s} event: ratio={r_ev:.3f} p={p_ev:.4f}   |   risk: ratio={r_risk:.3f} p={p_risk:.4f}")


if __name__ == "__main__":
    main()
