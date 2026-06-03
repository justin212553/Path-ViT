"""
CAMELYON17 패치 레벨 평가 스크립트
- WSI 1장의 모든 패치를 transformer에 한 번에 넣어 per-patch 분류
- 체크포인트에 저장된 Youden's J threshold 사용
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from data.patch_dataset import CAMELYON17NodeDataset
from models import PatchViT
from utils.metrics import compute_patch_metrics


def _encode_patches_chunked(
    cnn: nn.Module, patches: torch.Tensor, chunk_size: int, device: torch.device
) -> torch.Tensor:
    """CNN을 chunk_size 단위로 CPU→GPU 이동하며 실행 (대형 WSI OOM 방지)."""
    return torch.cat([
        cnn(patches[i : i + chunk_size].to(device, non_blocking=True))
        for i in range(0, patches.shape[0], chunk_size)
    ])


def evaluate_patch_level(
    checkpoint: str,
    cfg: Config | None = None,
    split: str = "eval",
    save_vis: bool = False,
    vis_dir: str = "heatmaps",
):
    if cfg is None:
        cfg = Config()
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    dataset = CAMELYON17NodeDataset(
        cfg.data, split=split,
        val_ratio=cfg.data.val_ratio, eval_ratio=cfg.data.eval_ratio,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = PatchViT(cfg.model).to(device)
    ckpt  = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    threshold = ckpt.get("threshold", 0.5)
    print(f"  (checkpoint threshold: {threshold:.4f})")
    model.eval()

    chunk_size = cfg.train.cnn_chunk_size
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            patches      = batch["patches"].squeeze(0)                              # (N, 3, H, W) — CPU 유지
            coords       = batch["coords"].squeeze(0).to(device, non_blocking=True) # (N, 2)
            patch_labels = batch["patch_labels"].squeeze(0).numpy()                 # (N,)
            slide_id     = batch["slide_id"][0]

            coords[:, 0] -= coords[:, 0].min()
            coords[:, 1] -= coords[:, 1].min()

            patch_tokens = _encode_patches_chunked(model.cnn, patches, chunk_size, device)
            ctx_tokens   = model.vit(patch_tokens, coords)
            patch_logits = model.classifier(ctx_tokens)

            scores = torch.softmax(patch_logits, dim=-1)[:, 1].float().cpu().numpy()
            preds  = (scores >= threshold).astype(np.int64)
            acc    = float((preds == patch_labels).mean())
            print(f"  {slide_id}: {int(patch_labels.sum())}/{len(patch_labels)} tumor patches  acc={acc:.3f}")

            all_scores.append(scores)
            all_labels.append(patch_labels)

            if save_vis:
                from utils.visualize import save_dual_overlay
                save_dual_overlay(
                    scores=scores,
                    coords=batch["coords"].squeeze(0).numpy(),
                    patch_labels=patch_labels,
                    slide_id=slide_id,
                    out_dir=vis_dir,
                )

    metrics = compute_patch_metrics(
        np.concatenate(all_scores),
        np.concatenate(all_labels),
    )
    print("\n=== Patch-Level Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--split", type=str, default="eval",
                        choices=["eval", "val", "all"],
                        help="평가에 사용할 split (기본: eval — held-out 셋)")
    parser.add_argument("--vis", action="store_true",
                        help="듀얼 오버레이 시각화 저장")
    parser.add_argument("--vis-dir", type=str, default="heatmaps",
                        help="시각화 저장 디렉토리")
    args = parser.parse_args()

    evaluate_patch_level(args.checkpoint, split=args.split, save_vis=args.vis, vis_dir=args.vis_dir)
