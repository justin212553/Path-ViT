"""
CAMELYON17 평가 스크립트
- 슬라이드 레벨 분류 (evaluate)
- 패치 레벨 분류  (evaluate_patch_level)
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data.patch_dataset import CAMELYON17PatchDataset, CAMELYON17NodeDataset
from models import PatchViT
from utils.metrics import compute_all_metrics, compute_patch_metrics


def evaluate(checkpoint: str, cfg: Config | None = None):
    if cfg is None:
        cfg = Config()
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    dataset = CAMELYON17PatchDataset(cfg.data, split="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = PatchViT(cfg.model).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            patches = batch["patches"].squeeze(0).to(device)
            coords  = batch["coords"].squeeze(0).to(device)
            out     = model(patches, coords)
            score   = torch.softmax(out["slide_logits"], dim=-1)[0, 1].item()
            all_scores.append(score)
            all_labels.append(batch["label"].item())

    metrics = compute_all_metrics(
        np.array(all_scores), np.array(all_labels), k=20, target_fpr=0.1
    )
    print("=== Slide-Level Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def evaluate_patch_level(
    checkpoint: str,
    cfg: Config | None = None,
    save_vis: bool = False,
    vis_dir: str = "heatmaps",
):
    """
    패치 레벨 평가: 각 패치를 독립적으로 추론 후 GT 패치 라벨과 비교.

    score >= 0.5 → pred=1(종양), 아니면 pred=0(정상)
    CAMELYON17NodeDataset의 patch_index.csv에서 GT 패치 라벨을 읽음.
    """
    if cfg is None:
        cfg = Config()
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    dataset = CAMELYON17NodeDataset(cfg.data)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    model = PatchViT(cfg.model).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    all_scores, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            patches      = batch["patches"].squeeze(0).to(device)    # (N, 3, H, W)
            coords       = batch["coords"].squeeze(0)                 # (N, 2) CPU
            patch_labels = batch["patch_labels"].squeeze(0).numpy()   # (N,)
            slide_id     = batch["slide_id"][0]

            N = patches.shape[0]
            slide_scores = np.empty(N, dtype=np.float32)

            for i in range(N):
                out = model(patches[i:i+1], coords[i:i+1].to(device))
                slide_scores[i] = torch.softmax(out["slide_logits"], dim=-1)[0, 1].item()

            slide_preds = (slide_scores >= 0.5).astype(np.int64)
            acc = float((slide_preds == patch_labels).mean())
            print(f"  {slide_id}: {int(patch_labels.sum())}/{N} tumor patches  acc={acc:.3f}")

            all_scores.append(slide_scores)
            all_preds.append(slide_preds)
            all_labels.append(patch_labels)

            if save_vis:
                from utils.visualize import save_dual_overlay
                save_dual_overlay(
                    scores=slide_scores,
                    coords=coords.numpy(),
                    patch_labels=patch_labels,
                    slide_id=slide_id,
                    out_dir=vis_dir,
                )

    all_scores = np.concatenate(all_scores)
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = compute_patch_metrics(all_scores, all_preds, all_labels)
    print("\n=== Patch-Level Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--patch-level", action="store_true",
                        help="패치 단위 평가 (CAMELYON17NodeDataset 사용)")
    parser.add_argument("--vis", action="store_true",
                        help="패치 레벨 평가 시 듀얼 오버레이 시각화 저장")
    parser.add_argument("--vis-dir", type=str, default="heatmaps",
                        help="시각화 저장 디렉토리")
    args = parser.parse_args()

    if args.patch_level:
        evaluate_patch_level(args.checkpoint, save_vis=args.vis, vis_dir=args.vis_dir)
    else:
        evaluate(args.checkpoint)
