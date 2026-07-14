"""
TCGA-PAAD / CPTAC-PDA WSI 생존(OS) 예측 학습 스크립트
태스크: 환자(case) 단위 OS(overall survival) risk score 회귀 — Cox Proportional Hazards
배치:   환자 1명이 보유한 모든 슬라이드 임베딩을 평균 풀링해 risk score 1개 산출.
        Cox loss는 위험집합(risk set) 비교를 위해 여러 환자를 한 minibatch(cox_batch_size)로
        묶어야 하므로, 그 minibatch가 찰 때마다 backward + optimizer.step()을 수행한다.
손실:   Cox partial negative log-likelihood (utils/losses.py::cox_ph_loss)
데이터: WSISurvivalDataset (data/dataset.py, --dataset {tcga,cptac})

검증:   단일 seed 기반 deterministic split — cfg.data.n_folds개의 stratified fold 중
        --fold(기본 0)번을 val로, 나머지를 train으로 고정해서 쓴다(data/dataset.py 참조).
        k-fold 전체를 순회하는 CV는 하지 않는다 — 파이프라인/아키텍처가 아직 바뀔 여지가
        많은 단계에서는 학습 1회당 비용이 커지는 k-fold보다 빠르게 반복 확인하는 쪽이 우선.
지표:   c-index, hazard ratio(HR), log-rank p-value (utils/metrics.py::compute_survival_metrics).
        HR/log-rank p는 risk score 중앙값으로 저위험/고위험군을 나눠 계산한다.
"""
import argparse
import math
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import Config
from data.dataset import WSISurvivalDataset
from models import PatchViT, LateFusionViT
from data.fit_clusters import CENTROIDS_DIR
from utils import load_env, send_slack
from utils.losses import cox_ph_loss
from utils.metrics import compute_survival_metrics


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


def _identity_collate(batch: list) -> list:
    """batch_size=1 전제 — DataLoader가 환자 1명의 슬라이드 리스트를 그대로 통과시키도록 함."""
    return batch[0]


def _patient_risk(model, patient_slides, device, amp_ctx, transform, chunk_size) -> torch.Tensor:
    """환자 1명이 보유한 슬라이드 전부를 forward해 임베딩을 평균 풀링한 뒤 risk score(scalar)를 계산한다."""
    with amp_ctx:
        slide_embeds = []
        for slide in patient_slides:
            coords = slide["coords"].to(device, non_blocking=True)
            if "features" in slide:
                out = model(coords, features=slide["features"])
            else:
                out = model(coords, patch_paths=slide["patch_paths"],
                             transform=transform, chunk_size=chunk_size)
            slide_embeds.append(out["embed"])

        patient_embed = torch.stack(slide_embeds).mean(dim=0)      # (D,) 또는 (2D,) — 슬라이드 평균 풀링
        risk = model.risk_head(patient_embed.unsqueeze(0)).view(1)  # (1,)
    return risk


def train_one_epoch(
    model, loader, optimizer, cfg, device, amp_ctx, transform
) -> float:
    model.train()
    if model.cnn.backbone is not None:
        model.cnn.backbone.eval()  # frozen backbone의 BN을 population stats(eval)로 고정 — train/eval 분포 불일치 방지
    total_loss    = 0.0
    total_batches = 0
    chunk_size    = cfg.train.cnn_chunk_size
    batch_size    = cfg.train.cox_batch_size

    risks, times, events = [], [], []

    def _flush():
        nonlocal risks, times, events, total_loss, total_batches
        if not risks:
            return
        risk_t  = torch.cat(risks)
        time_t  = torch.cat(times).to(device)
        event_t = torch.cat(events).to(device)

        loss = cox_ph_loss(risk_t, time_t, event_t)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item()
        total_batches += 1
        risks, times, events = [], [], []

    for patient_slides in loader:                # 환자 1명 분량의 슬라이드 리스트
        if len(patient_slides) == 0:
            continue
        risk = _patient_risk(model, patient_slides, device, amp_ctx, transform, chunk_size)

        risks.append(risk)
        times.append(patient_slides[0]["OS_time"])
        events.append(patient_slides[0]["OS_event"])

        if len(risks) >= batch_size:
            _flush()

    _flush()  # 마지막 남은 partial batch

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, cfg, device, amp_ctx, transform) -> dict:
    model.eval()
    all_risks, all_times, all_events = [], [], []
    chunk_size = cfg.train.cnn_chunk_size

    for patient_slides in loader:
        if len(patient_slides) == 0:
            continue
        risk = _patient_risk(model, patient_slides, device, amp_ctx, transform, chunk_size)

        all_risks.append(risk.float().item())
        all_times.append(float(patient_slides[0]["OS_time"].item()))
        all_events.append(int(patient_slides[0]["OS_event"].item()))

    risks  = np.array(all_risks)
    times  = np.array(all_times)
    events = np.array(all_events)
    return {
        **compute_survival_metrics(risks, times, events),
        "risks": risks, "times": times, "events": events,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="cptac", choices=["tcga", "cptac"],
        help="OS 예측에 사용할 데이터셋 (기본: cptac)",
    )
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
    parser.add_argument(
        "--fold", type=int, default=0,
        help="검증(val)으로 쓸 stratified fold 번호 (기본: 0) — seed가 같으면 항상 동일한 split",
    )
    parser.add_argument(
        "--n-folds", type=int, default=None,
        help="fold를 나눌 때의 총 fold 수 (기본: config.py의 DataConfig.n_folds)",
    )
    return parser.parse_args()


def main():
    load_env()
    args   = _parse_args()
    cfg    = Config()
    cfg.data.precomputed = not args.image
    if args.n_folds is not None:
        cfg.data.n_folds = args.n_folds

    # [LateFusion] --fusion 플래그 시 cluster_centroids.pt 로드 검증
    if args.fusion and not cfg.data.precomputed:
        raise ValueError("--fusion은 precomputed(features.pt) 모드에서만 지원됩니다. --image와 함께 사용 불가.")
    centroids_path = Path(__file__).parent / CENTROIDS_DIR
    if args.fusion and not centroids_path.exists():
        raise FileNotFoundError(
            f"cluster_centroids.pt 없음: {centroids_path}\n"
            "  먼저 실행: python -m data.fit_clusters"
        )
    cluster_centroids = torch.load(centroids_path, map_location="cpu") if args.fusion else None
    set_seed(cfg.train.seed)
    start_time = datetime.now()
    device = torch.device(cfg.train.device)

    torch.backends.cudnn.benchmark = True

    amp_ctx = _make_amp_ctx()

    if WANDB_AVAILABLE:
        model_prefix = "M1_C" if args.fusion else "M1"
        run_name = f"{args.dataset.upper()}_{model_prefix}_" + datetime.now().strftime("%m%d::%H%M")
        wandb.init(
            project="Path-ViT",
            name=run_name,
            config={
                "fold":                  args.fold,
                "n_folds":               cfg.data.n_folds,
                "epochs":                cfg.train.epochs,
                "lr":                    cfg.train.lr,
                "weight_decay":          cfg.train.weight_decay,
                "seed":                  cfg.train.seed,
                "warmup_epochs":         cfg.train.warmup_epochs,
                "cnn_chunk_size":        cfg.train.cnn_chunk_size,
                "cox_batch_size":        cfg.train.cox_batch_size,
                "embed_dim":             cfg.model.embed_dim,
                "num_heads":             cfg.model.num_heads,
                "num_transformer_layers":cfg.model.num_transformer_layers,
                "dropout":               cfg.model.dropout,
                "num_landmarks":         cfg.model.num_landmarks,
                # [LateFusion] 모델 종류 및 군집 수 기록 — ablation 비교용
                "model":                 "LateFusionViT" if args.fusion else "PatchViT",
                "num_clusters":          int(cluster_centroids.shape[0]) if args.fusion else 0,
                "dataset":               args.dataset,
            },
        )

    train_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="train",
                                   fold=args.fold, n_folds=cfg.data.n_folds)
    val_ds   = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="val",
                                   fold=args.fold, n_folds=cfg.data.n_folds)

    dl_kwargs = dict(
        batch_size=1,
        collate_fn=_identity_collate,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.data.num_workers > 0),
        prefetch_factor=2 if cfg.data.num_workers > 0 else None,
    )
    train_loader      = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **dl_kwargs)
    val_loader        = DataLoader(val_ds,   shuffle=False, **dl_kwargs)

    # [LateFusion] --fusion 플래그에 따라 모델 선택
    # PatchViT    : 기존 ViT+ABMIL 단일 경로 (ablation baseline)
    # LateFusionViT: ViT+ABMIL (Path A) + Cluster Histogram (Path B) Late Fusion
    if args.fusion:
        model = LateFusionViT(cfg.model, cluster_centroids, precomputed=cfg.data.precomputed).to(device)
    else:
        model = PatchViT(cfg.model, precomputed=cfg.data.precomputed).to(device)
    if model.cnn.backbone is not None:
        model.cnn.backbone.requires_grad_(False)

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
    print(f"Dataset: {args.dataset}  Fold: {args.fold}/{cfg.data.n_folds}  "
          f"Train: {len(train_ds)} patients  Val: {len(val_ds)} patients")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"AMP=bfloat16 | batch={cfg.train.cox_batch_size} patients (Cox risk set 단위) "
        f"| cnn_chunk={cfg.train.cnn_chunk_size} | workers={cfg.data.num_workers}"
    )
    ckpt_dir  = Path(__file__).parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # [LateFusion] 모델 종류별로 별도 checkpoint 저장 — ablation 결과 보존
    ckpt_path = ckpt_dir / (
        f"survival_{args.dataset}_best_fusion.pt" if args.fusion else f"survival_{args.dataset}_best.pt"
    )

    best_score   = -1.0
    best_metrics = {}
    for epoch in range(cfg.train.epochs):
        lr_now        = optimizer.param_groups[0]["lr"]
        loss          = train_one_epoch(model, train_loader, optimizer, cfg, device, amp_ctx, train_ds.transform)
        train_metrics = evaluate(model, train_eval_loader, cfg, device, amp_ctx, train_ds.transform)
        metrics       = evaluate(model, val_loader, cfg, device, amp_ctx, val_ds.transform)
        scheduler.step()

        c_index = metrics.get("c_index", float("nan"))
        score   = c_index if not math.isnan(c_index) else -1.0
        print(
            f"Epoch {epoch+1:3d} | lr={lr_now:.2e} | loss={loss:.4f} | "
            f"train_c_index={train_metrics['c_index']:.4f} | "
            f"val_c_index={metrics['c_index']:.4f} | "
            f"val_HR={metrics['hr']:.3f} | val_logrank_p={metrics['log_rank_p']:.4f}"
        )

        if WANDB_AVAILABLE:
            log_dict = {
                "train/loss":              loss,
                "train/lr":                lr_now,
                "train/c_index":           train_metrics["c_index"],
                "train/hr":                train_metrics["hr"],
                "train/log_rank_p":        train_metrics["log_rank_p"],
                "val_performance/c_index":    metrics["c_index"],
                "val_performance/hr":         metrics["hr"],
                "val_performance/log_rank_p": metrics["log_rank_p"],
            }
            wandb.log(log_dict, step=epoch + 1)

        if score > best_score:
            best_score   = score
            best_metrics = {"hr": metrics["hr"], "log_rank_p": metrics["log_rank_p"], "epoch": epoch + 1}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch":            epoch + 1,
                    "val_c_index":      best_score,
                    "val_hr":           metrics["hr"],
                    "val_log_rank_p":   metrics["log_rank_p"],
                },
                ckpt_path,
            )
            print(f"  → checkpoint saved (c_index={best_score:.4f}, HR={metrics['hr']:.3f}, "
                  f"log-rank p={metrics['log_rank_p']:.4f})")
            if WANDB_AVAILABLE:
                wandb.run.summary["best_val_c_index"]  = best_score
                wandb.run.summary["best_val_hr"]        = metrics["hr"]
                wandb.run.summary["best_val_log_rank_p"] = metrics["log_rank_p"]
                wandb.run.summary["best_epoch"]         = epoch + 1

    if WANDB_AVAILABLE:
        wandb.finish()

    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    send_slack(
        f":white_check_mark: *Path-ViT ({args.dataset.upper()} OS) 학습 완료*\n"
        f"> Epochs: {cfg.train.epochs} (best={best_metrics.get('epoch', '-')}) | "
        f"Best val C-index: *{best_score:.4f}* | HR: {best_metrics.get('hr', float('nan')):.3f} | "
        f"log-rank p: {best_metrics.get('log_rank_p', float('nan')):.4f}\n"
        f"> 소요 시간: {h}h {m}m {s}s"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        load_env()
        send_slack(f":x: *Path-ViT (OS) 학습 에러*\n```{type(e).__name__}: {e}```")
        raise
