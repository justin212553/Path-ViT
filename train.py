"""
CAMELYON17 WSI(노드) 단위 MIL 학습 스크립트
태스크: stage_labels.csv 기반 WSI 단위 이진 분류 (정상 / 전이)
손실:   CrossEntropyLoss (class-weighted, 클래스 불균형 보정)
데이터: CAMELYON17PatchDataset (patches_train 노드 폴더 + stage_labels.csv 라벨)
"""
import json
import math
import os
import random
import contextlib
import urllib.request
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import Config
from data.patch_dataset import CAMELYON17PatchDataset
from models import PatchViT
from utils.metrics import compute_patch_metrics


def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def send_slack(message: str):
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return
    try:
        data = json.dumps({"text": message}).encode()
        req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[Slack] 알림 전송 실패: {e}")


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
    cnn: nn.Module, patches: torch.Tensor, chunk_size: int, device: torch.device
) -> torch.Tensor:
    """CNN을 chunk_size 단위로 CPU→GPU 이동하며 실행 (대형 WSI OOM 방지)."""
    return torch.cat([
        cnn(patches[i : i + chunk_size].to(device, non_blocking=True))
        for i in range(0, patches.shape[0], chunk_size)
    ])


def _compute_class_weights(dataset: CAMELYON17PatchDataset, device) -> torch.Tensor:
    """훈련 셋 WSI 라벨 분포로 class weight 계산."""
    labels = dataset.items["label"].values
    n_neg = int((labels == 0).sum())
    n_pos = int((labels == 1).sum())
    total = n_neg + n_pos
    print(f"  Train slides: {n_neg} neg / {n_pos} pos  (pos ratio={n_pos/total:.3f})")
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
        patches = batch["patches"].squeeze(0)                              # (N, 3, H, W) — CPU 유지
        coords  = batch["coords"].squeeze(0).to(device, non_blocking=True) # (N, 2)
        label   = batch["label"].to(device, non_blocking=True)             # (1,)

        # 좌표를 0-기반으로 정규화
        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()

        with amp_ctx:
            patch_tokens          = _encode_patches_chunked(model.cnn, patches, chunk_size, device)  # (N, D)
            ctx_tokens            = model.vit(patch_tokens, coords)                                  # (N, D)
            wsi_embed, _          = model.attn_pool(ctx_tokens)                                      # (D,)
            wsi_logits            = model.classifier(wsi_embed.unsqueeze(0))                         # (1, 2)
            loss = criterion(wsi_logits, label) / accum_n

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
    all_scores, all_labels = [], []
    chunk_size = cfg.train.cnn_chunk_size

    for batch in loader:
        patches = batch["patches"].squeeze(0)                              # (N, 3, H, W) — CPU 유지
        coords  = batch["coords"].squeeze(0).to(device, non_blocking=True)
        label   = int(batch["label"].item())

        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()

        with amp_ctx:
            patch_tokens = _encode_patches_chunked(model.cnn, patches, chunk_size, device)
            ctx_tokens   = model.vit(patch_tokens, coords)
            wsi_embed, _ = model.attn_pool(ctx_tokens)
            wsi_logits   = model.classifier(wsi_embed.unsqueeze(0))

        score = torch.softmax(wsi_logits, dim=-1)[0, 1].float().item()

        all_scores.append(score)
        all_labels.append(label)

    return compute_patch_metrics(
        np.array(all_scores),
        np.array(all_labels),
    )


def main():
    _load_env()
    cfg    = Config()
    set_seed(cfg.train.seed)
    start_time = datetime.now()
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    amp_dtype = _get_amp_dtype(cfg.train.amp_dtype)
    amp_ctx   = _make_amp_ctx(amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    if WANDB_AVAILABLE:
        run_name = datetime.now().strftime("%m%d::%H%M")
        wandb.init(
            project="Path-ViT",
            name=run_name,
            config={
                "epochs":                cfg.train.epochs,
                "lr":                    cfg.train.lr,
                "weight_decay":          cfg.train.weight_decay,
                "seed":                  cfg.train.seed,
                "accumulate_grad_steps": cfg.train.accumulate_grad_steps,
                "warmup_epochs":         cfg.train.warmup_epochs,
                "amp_dtype":             cfg.train.amp_dtype,
                "cnn_chunk_size":        cfg.train.cnn_chunk_size,
                "embed_dim":             cfg.model.embed_dim,
                "num_heads":             cfg.model.num_heads,
                "num_transformer_layers":cfg.model.num_transformer_layers,
                "dropout":               cfg.model.dropout,
                "max_grid_size":         cfg.model.max_grid_size,
                "val_ratio":             cfg.data.val_ratio,
                "eval_ratio":            cfg.data.eval_ratio,
                "max_patches":           cfg.data.max_patches,
            },
        )

    train_ds = CAMELYON17PatchDataset(cfg.data, split="train", max_patches=cfg.data.max_patches)
    val_ds   = CAMELYON17PatchDataset(cfg.data, split="val",   max_patches=cfg.data.max_patches)
    eval_ds  = CAMELYON17PatchDataset(cfg.data, split="test",  max_patches=cfg.data.max_patches)

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

        if WANDB_AVAILABLE:
            wandb.log({
                "train/loss":      loss,
                "train/lr":        lr_now,
                "val/accuracy":    metrics["accuracy"],
                "val/auc_roc":     metrics["auc_roc"],
                "val/f1":          metrics["f1"],
                "val/precision":   metrics["precision"],
                "val/recall":      metrics["recall"],
            }, step=epoch + 1)

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "threshold":        metrics["threshold"],
                    "epoch":            epoch + 1,
                    "val_auc":          best_score,
                },
                ckpt_path,
            )
            print(f"  → checkpoint saved (auc={best_score:.4f}, threshold={metrics['threshold']:.4f})")
            if WANDB_AVAILABLE:
                wandb.run.summary["best_val_auc"]   = best_score
                wandb.run.summary["best_epoch"]      = epoch + 1
                wandb.run.summary["best_threshold"]  = metrics["threshold"]

    print("\n=== Final Evaluation on held-out eval set (best checkpoint) ===")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  (threshold from best checkpoint: {ckpt['threshold']:.4f})")
    else:
        print("WARNING: no checkpoint saved, using final model weights")
    final_metrics = evaluate(model, eval_loader, cfg, device, amp_ctx)
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    if WANDB_AVAILABLE:
        wandb.log({f"eval/{k}": v for k, v in final_metrics.items()})
        wandb.finish()

    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    send_slack(
        f":white_check_mark: *Path-ViT 학습 완료*\n"
        f"> Epochs: {cfg.train.epochs} | Best val AUC: *{best_score:.4f}*\n"
        f"> Eval AUC: *{final_metrics.get('auc_roc', 0):.4f}*  "
        f"F1: {final_metrics.get('f1', 0):.4f}  "
        f"Acc: {final_metrics.get('accuracy', 0):.4f}\n"
        f"> 소요 시간: {h}h {m}m {s}s"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _load_env()
        send_slack(f":x: *Path-ViT 학습 에러*\n```{type(e).__name__}: {e}```")
        raise
