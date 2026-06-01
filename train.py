"""
CAMELYON17 학습 스크립트 (순수 MIL)
태스크: 림프절 WSI → 슬라이드 레벨 전이 분류
손실:  CrossEntropy(slide_logits, slide_label)
"""
import math
import random
import contextlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from config import Config
from data.patch_dataset import CAMELYON17PatchDataset
from models import PatchViT
from utils.metrics import compute_all_metrics


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
    total   = cfg.train.epochs
    warmup  = cfg.train.warmup_epochs

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


def train_one_epoch(
    model, loader, optimizer, scaler, cfg, device, amp_ctx
) -> float:
    model.train()
    criterion  = nn.CrossEntropyLoss()
    total_loss = 0.0
    chunk_size = cfg.train.cnn_chunk_size
    accum_n    = cfg.train.accumulate_grad_steps

    optimizer.zero_grad()
    pending = 0  # 현재 accumulation 중인 WSI 수

    for step, batch in enumerate(loader):
        patches     = batch["patches"].squeeze(0).to(device, non_blocking=True)
        coords      = batch["coords"].squeeze(0).to(device, non_blocking=True)
        slide_label = batch["label"].to(device, non_blocking=True)

        with amp_ctx:
            patch_tokens = _encode_patches_chunked(model.cnn, patches, chunk_size)
            h_img, _     = model.vit(patch_tokens, coords)
            logits       = model.classifier(h_img)
            loss         = criterion(logits, slide_label) / accum_n

        scaler.scale(loss).backward()
        total_loss += loss.item() * accum_n
        pending    += 1

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
    all_scores, all_labels = [], []
    chunk_size = cfg.train.cnn_chunk_size

    for batch in loader:
        patches = batch["patches"].squeeze(0).to(device, non_blocking=True)
        coords  = batch["coords"].squeeze(0).to(device, non_blocking=True)

        with amp_ctx:
            patch_tokens = _encode_patches_chunked(model.cnn, patches, chunk_size)
            h_img, _     = model.vit(patch_tokens, coords)
            logits       = model.classifier(h_img)

        score = torch.softmax(logits, dim=-1)[0, 1].item()
        all_scores.append(score)
        all_labels.append(batch["label"].item())

    return compute_all_metrics(
        np.array(all_scores), np.array(all_labels), k=20, target_fpr=0.1
    )


def main():
    cfg    = Config()
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # 고정 입력 크기(패치)에 대해 cuDNN이 최적 커널 자동 선택
    torch.backends.cudnn.benchmark = True

    amp_dtype = _get_amp_dtype(cfg.train.amp_dtype)
    amp_ctx   = _make_amp_ctx(amp_dtype)
    # BF16은 FP32와 지수부 동일 → overflow 없음 → scaler 불필요
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    dl_kwargs = dict(
        batch_size=1,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),       # 페이지 고정 메모리로 DMA 가속
        persistent_workers=(cfg.data.num_workers > 0),  # 에폭 간 worker 재사용
        prefetch_factor=2 if cfg.data.num_workers > 0 else None,
    )

    train_ds     = CAMELYON17PatchDataset(cfg.data, split="train")
    val_ds       = CAMELYON17PatchDataset(cfg.data, split="val")
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)

    model     = PatchViT(cfg.model).to(device)
    model.cnn.backbone.requires_grad_(False)  # CNN backbone freeze: activation 저장 불필요
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = _build_scheduler(optimizer, cfg)

    dtype_name = str(amp_dtype).split(".")[-1] if amp_dtype else "fp32"
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"AMP={dtype_name} | accum={cfg.train.accumulate_grad_steps} WSIs "
        f"| cnn_chunk={cfg.train.cnn_chunk_size} | workers={cfg.data.num_workers}"
    )

    best_auc = 0.0
    for epoch in range(cfg.train.epochs):
        lr_now  = optimizer.param_groups[0]["lr"]
        loss    = train_one_epoch(model, train_loader, optimizer, scaler, cfg, device, amp_ctx)
        metrics = evaluate(model, val_loader, cfg, device, amp_ctx)
        scheduler.step()

        auc = metrics.get("sens@fpr10_auc_roc", 0.0)
        print(f"Epoch {epoch+1:3d} | lr={lr_now:.2e} | loss={loss:.4f} | {metrics}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "camelyon_best.pt")


if __name__ == "__main__":
    main()
