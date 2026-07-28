"""scripts/train_brca_m4.py 학습이 중간에 죽었을 때(예: CUDA OOM), 이미 저장된 best
체크포인트만으로 test 평가를 재현하기 위한 범용 스크립트.

사용 (PathViT-ray 환경):
  python -m scripts.eval_brca_m4_checkpoint --ckpt models/checkpoint/survival_brca_best_..._seed42.pt --knn-bias-attention
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from models.vit_pma import ViT_PMA
from models.rna_predictor import RNAPredictionHead
from models.clinical_encoder import age_stats_from_csv
from train import evaluate, _log_line
from utils.metrics import compute_time_dependent_auc
from scripts.brca_common import (
    CLINICAL_PATH, BRCASlideDataset, _identity_collate, load_case_table, load_rna_matrix, MANIFEST_PATH,
)
from contextlib import nullcontext

OUT_DIR = Path("data/brca_rna_gene_selection")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-genes", type=int, default=1500)
    p.add_argument("--no-spatial-embed", action="store_true")
    p.add_argument("--rel-bias-attention", action="store_true")
    p.add_argument("--knn-bias-attention", action="store_true")
    p.add_argument("--knn-k", type=int, default=8)
    return p.parse_args()


def main():
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config()
    cfg.data.seed = cfg.train.seed = args.seed
    if args.no_spatial_embed:
        cfg.model.use_spatial_embed = False
    if args.rel_bias_attention:
        cfg.model.use_rel_bias_attn = True
        cfg.model.use_nystrom = False
        cfg.model.use_spatial_embed = False
    if args.knn_bias_attention:
        cfg.model.use_knn_bias_attn = True
        cfg.model.knn_attn_k = args.knn_k
        cfg.model.use_nystrom = False
        cfg.model.use_spatial_embed = False

    gene_path = OUT_DIR / f"selected_genes_top_{args.n_genes}.csv"
    gene_ids = pd.read_csv(gene_path)["gene_id"].tolist()
    rna_input_dim = len(gene_ids)

    cases = load_case_table(args.seed)
    rna_df = load_rna_matrix(gene_ids)
    manifest = pd.read_csv(MANIFEST_PATH)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATH)

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    train_ds = BRCASlideDataset(cases[cases["split"] == "train"], rna_df, manifest)
    test_ds  = BRCASlideDataset(cases[cases["split"] == "test"],  rna_df, manifest)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **dl_kwargs)
    test_loader       = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    model = ViT_PMA(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=True, backbone="uni",
    ).to(device)
    model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, rna_input_dim).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"체크포인트 로드: {args.ckpt} (epoch={ckpt.get('epoch')}, val_c_index={ckpt.get('val_c_index')})")

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    train_metrics_final = evaluate(model, train_eval_loader, cfg, device, amp_ctx, None)
    test_metrics = evaluate(model, test_loader, cfg, device, amp_ctx, None)
    test_td_auc = compute_time_dependent_auc(
        train_metrics_final["times"], train_metrics_final["events"],
        test_metrics["times"], test_metrics["events"], test_metrics["risks"],
    )
    print(f"\n=== BRCA Internal Test (checkpoint epoch {ckpt.get('epoch')}) ===")
    print(_log_line("test", test_metrics, test_td_auc))


if __name__ == "__main__":
    main()
