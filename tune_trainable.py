"""
Ray Tune trainable 정의 모듈.

tune.py(드라이버 스크립트, __main__으로 실행됨)에 train_fn을 직접 정의하면
cloudpickle이 __main__ 모듈 함수를 "by value"로 직렬화해야 하는데, 이 과정에서
torch.backends.cudnn의 non-picklable 네이티브 핸들까지 끌려들어가
"TypeError: cannot pickle 'CudnnModule' object"가 발생한다.
trainable을 별도의 importable 모듈로 분리하면 cloudpickle이 "by reference"로
(모듈명 + 함수명만 저장) 직렬화할 수 있어 이 문제를 피할 수 있다.
"""
import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ray import tune

from config import Config
from data.patch_dataset import CAMELYON17NodeDataset
from models import PatchViT
from train import (
    _build_scheduler,
    _compute_class_weights,
    _get_amp_dtype,
    _identity_collate,
    _make_amp_ctx,
    evaluate,
    set_seed,
    train_one_epoch,
)

# embed_dim은 SpatialPositionEmbedding에서 2등분, ViTEncoder에서 num_heads로 나눠지므로
# 나눗셈이 항상 성립하는 (embed_dim, num_heads) 조합만 탐색 대상으로 삼는다.
EMBED_HEAD_CHOICES = [
    (128, 4), (128, 8),
    (256, 4), (256, 8),
    (384, 8), (384, 12),
]

# Ray Tune은 trial마다 작업 디렉터리를 trial별 결과 폴더로 바꾸므로,
# config.py의 patches_root/csv_path 같은 상대 경로는 그대로 두면 깨진다.
# 이 모듈(tune_trainable.py)이 위치한 프로젝트 루트를 기준으로 절대 경로화한다.
PROJECT_ROOT = Path(__file__).resolve().parent


def _build_cfg(base_cfg: Config, search_cfg: dict, tune_epochs: int) -> Config:
    cfg = copy.deepcopy(base_cfg)
    embed_dim, num_heads = search_cfg["embed_head"]
    cfg.model.embed_dim              = embed_dim
    cfg.model.num_heads              = num_heads
    cfg.model.num_transformer_layers = search_cfg["num_transformer_layers"]
    cfg.model.dropout                = search_cfg["dropout"]
    cfg.train.lr                     = search_cfg["lr"]
    cfg.train.weight_decay           = search_cfg["weight_decay"]
    cfg.train.warmup_epochs          = search_cfg["warmup_epochs"]
    cfg.train.epochs                 = tune_epochs
    cfg.data.patches_root            = str(PROJECT_ROOT / cfg.data.patches_root)
    cfg.data.csv_path                = str(PROJECT_ROOT / cfg.data.csv_path)
    return cfg


def train_fn(search_cfg: dict, base_cfg: Config, tune_epochs: int):
    cfg = _build_cfg(base_cfg, search_cfg, tune_epochs)
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    amp_dtype = _get_amp_dtype(cfg.train.amp_dtype)
    amp_ctx   = _make_amp_ctx(amp_dtype)
    scaler    = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    train_ds = CAMELYON17NodeDataset(cfg.data, split="train")
    val_ds   = CAMELYON17NodeDataset(cfg.data, split="val")

    dl_kwargs = dict(
        batch_size=1,
        collate_fn=_identity_collate,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.data.num_workers > 0),
        prefetch_factor=2 if cfg.data.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)

    model = PatchViT(cfg.model).to(device)
    model.cnn.backbone.requires_grad_(False)

    class_weights = _compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg)

    for epoch in range(cfg.train.epochs):
        loss    = train_one_epoch(model, train_loader, optimizer, scaler, cfg, device, amp_ctx, criterion, train_ds.transform)
        metrics = evaluate(model, val_loader, cfg, device, amp_ctx, val_ds.transform)
        scheduler.step()

        auc = metrics.get("auc_roc", 0.0)
        if math.isnan(auc):
            auc = 0.0

        tune.report({
            "val_auc_roc": auc,
            "val_f1":      metrics["f1"],
            "train_loss":  loss,
        })
