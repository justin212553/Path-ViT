"""
CAMELYON17 학습 스크립트 (순수 MIL)
태스크: 림프절 WSI → 슬라이드 레벨 전이 분류
손실:  CrossEntropy(slide_logits, slide_label)
"""
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from data.patch_dataset import CAMELYON17PatchDataset
from models import PatchViT
from utils.metrics import compute_all_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    criterion  = nn.CrossEntropyLoss()
    total_loss = 0.0

    for batch in loader:
        patches     = batch["patches"].squeeze(0).to(device)  # (N, 3, H, W)
        coords      = batch["coords"].squeeze(0).to(device)   # (N, 2)
        slide_label = batch["label"].to(device)               # (1,)

        optimizer.zero_grad()
        out  = model(patches, coords)
        loss = criterion(out["slide_logits"], slide_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        patches = batch["patches"].squeeze(0).to(device)
        coords  = batch["coords"].squeeze(0).to(device)
        out     = model(patches, coords)
        score   = torch.softmax(out["slide_logits"], dim=-1)[0, 1].item()
        all_scores.append(score)
        all_labels.append(batch["label"].item())

    return compute_all_metrics(
        np.array(all_scores), np.array(all_labels), k=20, target_fpr=0.1
    )


def main():
    cfg = Config()
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    train_ds = CAMELYON17PatchDataset(
        preprocessed_root=cfg.data.preprocessed_root,
        split="train",
        val_centers=cfg.data.val_centers,
    )
    val_ds = CAMELYON17PatchDataset(
        preprocessed_root=cfg.data.preprocessed_root,
        split="val",
        val_centers=cfg.data.val_centers,
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=cfg.data.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=cfg.data.num_workers)

    model = PatchViT(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    best_auc = 0.0
    for epoch in range(cfg.train.epochs):
        loss    = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        auc     = metrics.get("sens@fpr10_auc_roc", 0.0)
        print(f"Epoch {epoch+1:3d} | loss={loss:.4f} | {metrics}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "camelyon_best.pt")


if __name__ == "__main__":
    main()
