"""scripts/train_brca_spatial_residual.py 학습 도중 조기 중단 시, 저장된 best branch
체크포인트 + Stage1 precompute 캐시만으로 internal test 평가를 재현하기 위한 스크립트.

사용 (PathViT-ray 환경):
  python -m scripts.eval_brca_spatial_residual --layer-type gcn
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from models.spatial_residual import SpatialResidualBranch
from utils.metrics import compute_survival_metrics

DEFAULT_STAGE1_CKPT = "models/checkpoint/survival_brca_best_brca_pma_top1500_ss_aux_seed42.pt"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", type=str, default=DEFAULT_STAGE1_CKPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--edge-dropout", type=float, default=0.2)
    p.add_argument("--layer-type", type=str, default="gcn", choices=["gcn", "attention"])
    return p.parse_args()


def _branch_risk(branch: SpatialResidualBranch, patient: dict, device) -> torch.Tensor:
    slide_reprs = [
        branch.encode(pt.to(device, non_blocking=True), c.to(device, non_blocking=True))
        for pt, c in patient["slides"]
    ]
    pooled = torch.stack(slide_reprs).mean(dim=0)
    return branch.head(branch.head_drop(pooled)).view(())


@torch.no_grad()
def _evaluate(branch: SpatialResidualBranch, cache: list[dict], device) -> dict:
    branch.eval()
    risks, times, events = [], [], []
    for patient in cache:
        final_risk = patient["base_risk"].to(device) + _branch_risk(branch, patient, device)
        risks.append(final_risk.item())
        times.append(float(patient["OS_time"].item()))
        events.append(int(patient["OS_event"].item()))
    return compute_survival_metrics(np.array(risks), np.array(times), np.array(events))


def main():
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = Path("data/spatial_residual_cache")
    stage1_tag = Path(args.stage1_ckpt).stem
    cache_path = cache_dir / f"brca_seed{args.seed}_{stage1_tag}.pt"
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    test_cache = cached["test"]
    print(f"Stage1 precompute 캐시 로드: {cache_path} (test={len(test_cache)}명)")

    patch_dim = test_cache[0]["slides"][0][0].shape[-1]
    branch = SpatialResidualBranch(
        patch_dim=patch_dim, hidden_dim=args.hidden_dim, k=args.k,
        num_layers=args.num_layers, dropout=args.dropout, edge_dropout=args.edge_dropout,
        layer_type=args.layer_type,
    ).to(device)

    branch_ckpt_path = Path("models/checkpoint") / f"spatial_residual_brca_{args.layer_type}_seed{args.seed}_best.pt"
    best = torch.load(branch_ckpt_path, map_location=device, weights_only=False)
    branch.load_state_dict(best["branch_state_dict"])
    print(f"Branch 체크포인트 로드: {branch_ckpt_path} (epoch={best['epoch']}, val_c_index={best['val_c_index']:.4f})")

    test_metrics = _evaluate(branch, test_cache, device)
    print(f"\n=== BRCA Internal Test 성능 (best checkpoint, epoch {best['epoch']}) ===")
    print(f"test_c_index={test_metrics['c_index']:.4f} | test_HR={test_metrics['hr']:.3f} "
          f"| test_logrank_p={test_metrics['log_rank_p']:.4f}")


if __name__ == "__main__":
    main()
