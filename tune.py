"""
Ray Tune 기반 하이퍼파라미터 탐색 스크립트
탐색 대상: lr, weight_decay, warmup_epochs, dropout, (embed_dim, num_heads) 조합, num_transformer_layers
스케줄러:  ASHA — 성능이 낮은 trial을 조기 종료해 탐색 효율을 높임
지표:      val_auc_roc (maximize)

사용 예:
    python tune.py
(탐색 규모는 파일 하단의 NUM_SAMPLES / TUNE_EPOCHS / GPUS_PER_TRIAL / CPUS_PER_TRIAL 상수로 조절)
"""
import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ray import tune, train as ray_train
from ray.tune.schedulers import ASHAScheduler

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

SEARCH_SPACE = {
    "lr":                     tune.loguniform(1e-6, 1e-5),
    "weight_decay":           tune.loguniform(1e-5, 1e-4),
    "warmup_epochs":          tune.choice([1, 2, 3, 4, 5]),
    "dropout":                tune.uniform(0.0, 0.4),
    "embed_head":             tune.choice(EMBED_HEAD_CHOICES),
    "num_transformer_layers": tune.choice([2, 4, 6, 8]),
}


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
    return cfg


def train_fn(search_cfg: dict, base_cfg: Config, tune_epochs: int):
    cfg = _build_cfg(base_cfg, search_cfg, tune_epochs)
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    amp_dtype = _get_amp_dtype(cfg.train.amp_dtype)
    amp_ctx   = _make_amp_ctx(amp_dtype)
    scaler    = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    train_ds = CAMELYON17NodeDataset(cfg.data, split="train", max_patches=cfg.data.max_patches)
    val_ds   = CAMELYON17NodeDataset(cfg.data, split="val",   max_patches=cfg.data.max_patches)

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
        loss    = train_one_epoch(model, train_loader, optimizer, scaler, cfg, device, amp_ctx, criterion)
        metrics = evaluate(model, val_loader, cfg, device, amp_ctx)
        scheduler.step()

        auc = metrics.get("auc_roc", 0.0)
        if math.isnan(auc):
            auc = 0.0

        ray_train.report({
            "val_auc_roc": auc,
            "val_f1":      metrics["f1"],
            "train_loss":  loss,
        })


NUM_SAMPLES    = 20   # 탐색할 trial(하이퍼파라미터 조합) 수
TUNE_EPOCHS    = 8    # trial당 학습 epoch 수 (본 학습보다 짧게)
# 주의: 드라이버 프로세스에서 torch.cuda.is_available() 등 CUDA 관련 호출을 하면
# 드라이버에 CUDA 컨텍스트가 생성되어, 이후 Ray가 trial 함수를 pickle할 때
# torch.backends.cudnn의 non-picklable 네이티브 핸들까지 직렬화하려다 실패한다.
# (TypeError: cannot pickle 'CudnnModule' object) — CUDA 초기화는 각 trial 워커 안(train_fn)에서만 일어나야 함.
GPUS_PER_TRIAL = 1.0
CPUS_PER_TRIAL = 4.0


def main():
    base_cfg = Config()

    asha = ASHAScheduler(
        metric="val_auc_roc",
        mode="max",
        max_t=TUNE_EPOCHS,
        grace_period=2,
        reduction_factor=2,
    )

    trainable = tune.with_resources(
        tune.with_parameters(train_fn, base_cfg=base_cfg, tune_epochs=TUNE_EPOCHS),
        resources={"cpu": CPUS_PER_TRIAL, "gpu": GPUS_PER_TRIAL},
    )

    tuner = tune.Tuner(
        trainable,
        param_space=SEARCH_SPACE,
        tune_config=tune.TuneConfig(
            scheduler=asha,
            num_samples=NUM_SAMPLES,
        ),
        run_config=tune.RunConfig(
            name="path_vit_raytune",
            storage_path=str(Path(__file__).parent / "ray_results"),
        ),
    )

    results = tuner.fit()

    best = results.get_best_result(metric="val_auc_roc", mode="max")
    print("\n=== Best trial ===")
    print("Config:", best.config)
    print(f"val_auc_roc: {best.metrics['val_auc_roc']:.4f}")


if __name__ == "__main__":
    main()
