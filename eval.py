"""
CAMELYON17 평가 스크립트 (슬라이드 레벨 분류)
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data.patch_dataset import CAMELYON17PatchDataset
from models import PatchViT
from utils.metrics import compute_all_metrics


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
    print("=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    args = parser.parse_args()
    evaluate(args.checkpoint)
