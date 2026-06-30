"""
CAMELYON17 WSI(노드) 단위 MIL 학습 스크립트 (모델 파이프라인 점검용)
태스크: stage_labels.csv 기반 WSI 단위 이진 분류 (정상 / 전이) — 노드별 라벨/예측 유지
배치:   환자 1명 = 1 스텝. 그 환자가 가진 모든 노드를 누적(backward)한 뒤 한 번 optimizer.step()
손실:   CrossEntropyLoss (class-weighted, 클래스 불균형 보정)
데이터: CAMELYON17NodeDataset (patches_root 노드 폴더 + stage_labels.csv 라벨, train/val만 사용)
"""
import argparse
import json
import math
import os
import random
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
from data.patch_dataset import CAMELYON17NodeDataset
# from models import PatchViT                          # [LateFusion] 기존 단일 경로 모델
from models import PatchViT, LateFusionViT             # [LateFusion] Late Fusion 모델 추가
from data.fit_clusters import CENTROIDS_FILENAME       # [LateFusion] 군집 중심 파일명
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


def _make_amp_ctx() -> torch.autocast:
    """A30 전용 bfloat16 autocast — bf16은 fp32와 지수 범위가 같아 loss scaling이 불필요하다."""
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


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


def _compute_class_weights(dataset: CAMELYON17NodeDataset, device) -> torch.Tensor:
    """훈련 셋 노드 라벨 분포로 inverse-frequency class weight 계산."""
    labels = dataset.items["label"].values
    n_neg = int((labels == 0).sum())
    n_pos = int((labels == 1).sum())
    total = n_neg + n_pos
    print(f"  Train slides: {n_neg} neg / {n_pos} pos  (pos ratio={n_pos/total:.3f})")
    return torch.tensor(
        [total / (2.0 * n_neg), total / (2.0 * n_pos)],
        dtype=torch.float32, device=device,
    )


def _identity_collate(batch: list) -> list:
    """batch_size=1 전제 — DataLoader가 환자 1명의 노드 리스트를 그대로 통과시키도록 함."""
    return batch[0]


def train_one_epoch(
    model, loader, optimizer, cfg, device, amp_ctx, criterion, transform
) -> float:
    model.train()
    if model.cnn.backbone is not None:
        model.cnn.backbone.eval()  # frozen backbone의 BN을 population stats(eval)로 고정 — train/eval 분포 불일치 방지
    total_loss  = 0.0
    total_nodes = 0
    chunk_size  = cfg.train.cnn_chunk_size

    for patient_nodes in loader:                # 환자 1명 분량의 WSI 리스트
        n_nodes = len(patient_nodes)
        if n_nodes == 0:                        # 배경 필터링 등으로 패치가 없는 예외 처리 가드
            continue

        optimizer.zero_grad()
        patient_accumulated_loss = 0.0

        # 1. 한 환자의 모든 WSI(노드)를 순회하며 Gradient 누적
        for node in patient_nodes:
            coords = node["coords"].to(device, non_blocking=True)  # (N, 2)
            label  = node["label"].to(device, non_blocking=True)   # (1,)

            with amp_ctx:
                if "features" in node:
                    out = model(coords, features=node["features"])
                else:
                    out = model(coords, patch_paths=node["patch_paths"],
                                transform=transform, chunk_size=chunk_size)

                # 정석: 개별 WSI 로스를 구한 뒤, 환자가 가진 WSI 총 개수로 균등하게 나누어 줍니다.
                loss = criterion(out["wsi_logits"], label) / n_nodes

            loss.backward()

            # 로깅용 값 누적 (정규화 전 본래의 Loss 값 복원 기록)
            patient_accumulated_loss += loss.item() * n_nodes
            total_nodes += 1

        # 2. 환자 한 명의 모든 슬라이드 연산이 '완전히 끝난 후' 가중치 업데이트 실행
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += patient_accumulated_loss

    # 에포크 전체 평균 Loss 반환 (WSI 개수 기준 평균)
    return total_loss / max(total_nodes, 1)

@torch.no_grad()
def evaluate(model, loader, cfg, device, amp_ctx, transform) -> dict:
    model.eval()
    all_scores, all_labels = [], []
    chunk_size = cfg.train.cnn_chunk_size

    for patient_nodes in loader:
        for node in patient_nodes:
            coords = node["coords"].to(device, non_blocking=True)
            label  = int(node["label"].item())

            with amp_ctx:
                if "features" in node:
                    out = model(coords, features=node["features"])
                else:
                    out = model(coords, patch_paths=node["patch_paths"],
                                transform=transform, chunk_size=chunk_size)

            score = torch.softmax(out["wsi_logits"], dim=-1)[0, 1].float().item()

            all_scores.append(score)
            all_labels.append(label)

    scores = np.array(all_scores)
    labels = np.array(all_labels)
    return {
        **compute_patch_metrics(scores, labels),
        "scores": scores,
        "labels": labels,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", action="store_true",
        help="패치 jpg/png를 매 forward마다 ResNet50으로 직접 인코딩 (기본: data/extract_features.py로 "
             "사전 추출한 features.pt 사용)",
    )
    # [LateFusion] --fusion 플래그로 LateFusionViT 사용 여부 선택
    # 미지정 시 기존 PatchViT(ViT+ABMIL)로 동작 — ablation baseline 유지
    parser.add_argument(
        "--fusion", action="store_true",
        help="LateFusionViT 사용 (ViT+ABMIL + Cluster Histogram). "
             "data/fit_clusters.py 실행으로 cluster_centroids.pt 사전 생성 필요.",
    )
    return parser.parse_args()


def main():
    _load_env()
    args   = _parse_args()
    cfg    = Config()
    cfg.data.precomputed = not args.image

    # [LateFusion] --fusion 플래그 시 cluster_centroids.pt 로드 검증
    if args.fusion and not cfg.data.precomputed:
        raise ValueError("--fusion은 precomputed(features.pt) 모드에서만 지원됩니다. --image와 함께 사용 불가.")
    centroids_path = Path(__file__).parent / CENTROIDS_FILENAME
    if args.fusion and not centroids_path.exists():
        raise FileNotFoundError(
            f"cluster_centroids.pt 없음: {centroids_path}\n"
            "  먼저 실행: python -m data.fit_clusters"
        )
    set_seed(cfg.train.seed)
    start_time = datetime.now()
    device = torch.device(cfg.train.device)

    torch.backends.cudnn.benchmark = True

    amp_ctx = _make_amp_ctx()

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
                "warmup_epochs":         cfg.train.warmup_epochs,
                "cnn_chunk_size":        cfg.train.cnn_chunk_size,
                "embed_dim":             cfg.model.embed_dim,
                "num_heads":             cfg.model.num_heads,
                "num_transformer_layers":cfg.model.num_transformer_layers,
                "dropout":               cfg.model.dropout,
                "num_landmarks":         cfg.model.num_landmarks,
                # [LateFusion] 모델 종류 및 군집 수 기록 — ablation 비교용
                "model":                 "LateFusionViT" if args.fusion else "PatchViT",
                "num_clusters":          int(cluster_centroids.shape[0]) if args.fusion else 0,
            },
        )

    train_ds = CAMELYON17NodeDataset(cfg.data, split="train")
    val_ds   = CAMELYON17NodeDataset(cfg.data, split="val")

    dl_kwargs = dict(
        batch_size=1,
        collate_fn=_identity_collate,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.data.num_workers > 0),
        prefetch_factor=2 if cfg.data.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)

    # [LateFusion] --fusion 플래그에 따라 모델 선택
    # PatchViT    : 기존 ViT+ABMIL 단일 경로 (ablation baseline)
    # LateFusionViT: ViT+ABMIL (Path A) + Cluster Histogram (Path B) Late Fusion
    if args.fusion:
        cluster_centroids = torch.load(centroids_path, map_location="cpu")
        model = LateFusionViT(cfg.model, cluster_centroids, precomputed=cfg.data.precomputed).to(device)
    else:
        model = PatchViT(cfg.model, precomputed=cfg.data.precomputed).to(device)
    if model.cnn.backbone is not None:
        model.cnn.backbone.requires_grad_(False)

    class_weights = _compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  Class weights: neg={class_weights[0]:.3f}  pos={class_weights[1]:.3f}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg)

    mode = "precomputed features" if cfg.data.precomputed else "raw image (--image)"
    print(f"Mode: {mode}")
    # [LateFusion] 모델 종류 및 군집 수 출력
    if args.fusion:
        K = int(cluster_centroids.shape[0])
        print(f"Model: LateFusionViT (ViT+ABMIL + ClusterHistogram, K={K})")
    else:
        print(f"Model: PatchViT (ViT+ABMIL baseline)")
    print(f"Train: {len(train_ds)} patients  Val: {len(val_ds)} patients")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"AMP=bfloat16 | batch=1 patient (모든 노드 누적 후 1 step) "
        f"| cnn_chunk={cfg.train.cnn_chunk_size} | workers={cfg.data.num_workers}"
    )
    ckpt_dir  = Path(__file__).parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # [LateFusion] 모델 종류별로 별도 checkpoint 저장 — ablation 결과 보존
    # ckpt_path = ckpt_dir / "camelyon_best.pt"                          # [LateFusion] 기존 경로
    ckpt_path = ckpt_dir / ("camelyon_best_fusion.pt" if args.fusion else "camelyon_best.pt")

    best_score = 0.0
    for epoch in range(cfg.train.epochs):
        lr_now  = optimizer.param_groups[0]["lr"]
        loss    = train_one_epoch(model, train_loader, optimizer, cfg, device, amp_ctx, criterion, train_ds.transform)
        metrics = evaluate(model, val_loader, cfg, device, amp_ctx, val_ds.transform)
        scheduler.step()

        auc   = metrics.get("auc_roc", 0.0)
        score = auc if not math.isnan(auc) else metrics.get("f1", 0.0)
        print(
            f"Epoch {epoch+1:3d} | lr={lr_now:.2e} | loss={loss:.4f} | "
            f"acc={metrics['accuracy']:.4f}  auc={metrics['auc_roc']:.4f}  "
            f"f1={metrics['f1']:.4f}  prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}"
        )

        if WANDB_AVAILABLE:
            scores, labels = metrics["scores"], metrics["labels"]
            log_dict = {
                "train/loss":      loss,
                "train/lr":        lr_now,
                "val/accuracy":    metrics["accuracy"],
                "val/auc_roc":     metrics["auc_roc"],
                "val/f1":          metrics["f1"],
                "val/precision":   metrics["precision"],
                "val/recall":      metrics["recall"],
            }
            if (labels == 0).any():
                log_dict["val/score_hist_neg"] = wandb.Histogram(scores[labels == 0])
                log_dict["val/score_mean_neg"] = float(scores[labels == 0].mean())
            if (labels == 1).any():
                log_dict["val/score_hist_pos"] = wandb.Histogram(scores[labels == 1])
                log_dict["val/score_mean_pos"] = float(scores[labels == 1].mean())
            wandb.log(log_dict, step=epoch + 1)

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch":            epoch + 1,
                    "val_auc":          best_score,
                },
                ckpt_path,
            )
            print(f"  → checkpoint saved (auc={best_score:.4f})")
            if WANDB_AVAILABLE:
                wandb.run.summary["best_val_auc"] = best_score
                wandb.run.summary["best_epoch"]   = epoch + 1

    if WANDB_AVAILABLE:
        wandb.finish()

    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    send_slack(
        f":white_check_mark: *Path-ViT 학습 완료*\n"
        f"> Epochs: {cfg.train.epochs} | Best val AUC: *{best_score:.4f}*\n"
        f"> 소요 시간: {h}h {m}m {s}s"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _load_env()
        send_slack(f":x: *Path-ViT 학습 에러*\n```{type(e).__name__}: {e}```")
        raise
