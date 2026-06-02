"""
CAMELYON17 패치 레벨 학습 스크립트
태스크: annotation 기반 패치 이진 분류 (종양 / 정상)
손실:   CrossEntropyLoss (class-weighted, 클래스 불균형 보정)
데이터: CAMELYON17NodeDataset (eval_patch_index.csv, GT 패치 라벨 포함)
"""
import math
import random
import contextlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from config import Config
from data.patch_dataset import CAMELYON17NodeDataset
from models import PatchViT
from utils.metrics import compute_patch_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_amp_dtype(cfg_dtype: str) -> torch.dtype | None:
    """A30 → bfloat16, V100 → float16, "none" → AMP 비활성화."""
    if cfg_dtype == "none":
        return None
    if cfg_dtype == "bfloat16" or (
        cfg_dtype == "auto"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float16


def _make_amp_ctx(amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _build_scheduler(optimizer, cfg):
    """Linear warmup → cosine decay (epoch 단위)."""
    total  = cfg.train.epochs
    warmup = cfg.train.warmup_epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _encode_patches_chunked(
    cnn: nn.Module, patches: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """CNN을 chunk_size 단위 서브배치로 나눠 실행 (대형 WSI OOM 방지)."""
    return torch.cat([
        cnn(patches[i : i + chunk_size])
        for i in range(0, patches.shape[0], chunk_size)
    ])


def _compute_class_weights(dataset: CAMELYON17NodeDataset, device) -> torch.Tensor:
    """훈련 셋 패치 라벨 분포로 class weight 계산."""
    all_labels = np.concatenate([item["df"]["patch_label"].values for item in dataset.slides])
    n_neg = int((all_labels == 0).sum())
    n_pos = int((all_labels == 1).sum())
    total = n_neg + n_pos
    print(f"  Train patches: {n_neg} neg / {n_pos} pos  (pos ratio={n_pos/total:.3f})")
    return torch.tensor(
        [total / (2.0 * n_neg), total / (2.0 * n_pos)],
        dtype=torch.float32,
        device=device,
    )


def train_one_epoch(
    model, loader, optimizer, scaler, cfg, device, amp_ctx, criterion
) -> float:
    model.train()
    total_loss = 0.0
    chunk_size = cfg.train.cnn_chunk_size
    accum_n    = cfg.train.accumulate_grad_steps

    optimizer.zero_grad()
    pending = 0

    for step, batch in enumerate(loader):
        patches      = batch["patches"].squeeze(0).to(device, non_blocking=True)       # (N, 3, H, W)
        coords       = batch["coords"].squeeze(0).to(device, non_blocking=True)        # (N, 2)
        patch_labels = batch["patch_labels"].squeeze(0).to(device, non_blocking=True)  # (N,)

        # 좌표를 0-기반으로 정규화
        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()

        with amp_ctx:
            patch_tokens  = _encode_patches_chunked(model.cnn, patches, chunk_size)  # (N, D)
            _, all_tokens = model.vit(patch_tokens, coords)                           # (N+1, D)
            patch_logits  = model.classifier(all_tokens[1:])                         # (N, 2)  CLS 토큰 제외
            loss = criterion(patch_logits, patch_labels) / accum_n

        scaler.scale(loss).backward()
        total_loss += loss.item() * accum_n
        pending += 1

        last_step = (step == len(loader) - 1)
        if pending == accum_n or last_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            pending = 0

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, cfg, device, amp_ctx) -> dict:
    model.eval()
    all_scores, all_preds, all_labels = [], [], []
    chunk_size = cfg.train.cnn_chunk_size

    for batch in loader:
        patches      = batch["patches"].squeeze(0).to(device, non_blocking=True)
        coords       = batch["coords"].squeeze(0).to(device, non_blocking=True)
        patch_labels = batch["patch_labels"].squeeze(0).numpy()

        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()

        with amp_ctx:
            patch_tokens  = _encode_patches_chunked(model.cnn, patches, chunk_size)
            _, all_tokens = model.vit(patch_tokens, coords)
            patch_logits  = model.classifier(all_tokens[1:])

        scores = torch.softmax(patch_logits, dim=-1)[:, 1].float().cpu().numpy()
        preds  = (scores >= 0.5).astype(np.int64)

        all_scores.append(scores)
        all_preds.append(preds)
        all_labels.append(patch_labels)

    return compute_patch_metrics(
        np.concatenate(all_scores),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


def main():
    cfg    = Config()
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    amp_dtype = _get_amp_dtype(cfg.train.amp_dtype)
    amp_ctx   = _make_amp_ctx(amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    split_kwargs = dict(
        val_ratio=cfg.data.val_ratio, eval_ratio=cfg.data.eval_ratio, seed=cfg.train.seed
    )
    train_ds = CAMELYON17NodeDataset(cfg.data, split="train", **split_kwargs)
    val_ds   = CAMELYON17NodeDataset(cfg.data, split="val",   **split_kwargs)
    eval_ds  = CAMELYON17NodeDataset(cfg.data, split="eval",  **split_kwargs)

    dl_kwargs = dict(
        batch_size=1,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.data.num_workers > 0),
        prefetch_factor=2 if cfg.data.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    eval_loader  = DataLoader(eval_ds,  shuffle=False, **dl_kwargs)

    model = PatchViT(cfg.model).to(device)
    model.cnn.backbone.requires_grad_(False)

    class_weights = _compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg)

    dtype_name = str(amp_dtype).split(".")[-1] if amp_dtype else "fp32"
    print(f"Train: {len(train_ds)} slides  Val: {len(val_ds)} slides  Eval: {len(eval_ds)} slides")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"AMP={dtype_name} | accum={cfg.train.accumulate_grad_steps} slides "
        f"| cnn_chunk={cfg.train.cnn_chunk_size} | workers={cfg.data.num_workers}"
    )
    print(f"Class weights: neg={class_weights[0]:.3f}  pos={class_weights[1]:.3f}")

    ckpt_dir  = Path(__file__).parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "camelyon_best.pt"

    best_score = 0.0
    for epoch in range(cfg.train.epochs):
        lr_now  = optimizer.param_groups[0]["lr"]
        loss    = train_one_epoch(model, train_loader, optimizer, scaler, cfg, device, amp_ctx, criterion)
        metrics = evaluate(model, val_loader, cfg, device, amp_ctx)
        scheduler.step()

        auc   = metrics.get("auc_roc", 0.0)
        score = auc if not math.isnan(auc) else metrics.get("f1", 0.0)
        print(
            f"Epoch {epoch+1:3d} | lr={lr_now:.2e} | loss={loss:.4f} | "
            f"acc={metrics['accuracy']:.4f}  auc={metrics['auc_roc']:.4f}  "
            f"f1={metrics['f1']:.4f}  prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → checkpoint saved (auc={best_score:.4f})")

    print("\n=== Final Evaluation on held-out eval set (best checkpoint) ===")
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print("WARNING: no checkpoint saved, using final model weights")
    final_metrics = evaluate(model, eval_loader, cfg, device, amp_ctx)
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
