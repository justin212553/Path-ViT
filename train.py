"""
TCGA-PAAD / CPTAC-PDA WSI 생존(OS) 예측 학습 스크립트
태스크: 환자(case) 단위 OS(overall survival) risk score 회귀 — Cox Proportional Hazards
배치:   환자 1명이 보유한 모든 슬라이드 임베딩을 평균 풀링해 risk score 1개 산출.
        Cox loss는 위험집합(risk set) 비교를 위해 여러 환자를 한 minibatch(cox_batch_size)로
        묶어야 하므로, 그 minibatch가 찰 때마다 backward + optimizer.step()을 수행한다.
손실:   Cox partial negative log-likelihood (utils/losses.py::cox_ph_loss)
데이터: WSISurvivalDataset (data/dataset.py, --dataset {tcga,cptac})

검증:   stratified k-fold(cfg.data.n_folds). 코호트가 180명 안팎으로 작아 고정 단일
        train/val split은 val c-index 추정 분산이 크다 — 전체 환자를 한 번씩 val로
        순환시키는 fold마다 모델을 처음부터 새로 학습하고, 각 fold의 최고 체크포인트로
        얻은 val 예측을 모두 pooling해 코호트 전체에 대한 c-index(OOF c-index)를 최종
        지표로 삼는다. fold별 c-index의 평균±표준편차도 함께 참고용으로 보고한다.
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
        "--n-folds", type=int, default=None,
        help="stratified k-fold 수 (기본: config.py의 DataConfig.n_folds)",
    )
    return parser.parse_args()


def _run_fold(
    fold: int, cfg: Config, args: argparse.Namespace, cluster_centroids,
    device: torch.device, amp_ctx, ckpt_dir: Path, wandb_group: str,
) -> tuple[float, dict]:
    """
    fold 1개에 대해 모델을 처음부터 학습한다 (모델/optimizer/scheduler를 새로 생성).

    Returns:
        best_score: 이 fold에서 얻은 최고 val c-index (comparable pair가 없으면 -1.0)
        oof:        최고 체크포인트로 val set을 다시 평가한 결과 (risks/times/events 포함)
                    — fold 간 pooling으로 전체 코호트 OOF c-index를 구하는 데 사용
    """
    train_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="train",
                                   fold=fold, n_folds=cfg.data.n_folds)
    val_ds   = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="val",
                                   fold=fold, n_folds=cfg.data.n_folds)

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

    suffix = "_fusion" if args.fusion else ""
    ckpt_path = ckpt_dir / f"survival_{args.dataset}_fold{fold}_best{suffix}.pt"

    print(f"  Train: {len(train_ds)} patients  Val: {len(val_ds)} patients  "
          f"Params: {sum(p.numel() for p in model.parameters()):,}")

    if WANDB_AVAILABLE:
        wandb.init(
            project="Path-ViT",
            group=wandb_group,
            name=f"{wandb_group}_fold{fold}",
            reinit=True,
            config={
                "fold": fold, "n_folds": cfg.data.n_folds,
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
                "model":                 "LateFusionViT" if args.fusion else "PatchViT",
                "num_clusters":          int(cluster_centroids.shape[0]) if args.fusion else 0,
                "dataset":               args.dataset,
            },
        )

    best_score = -1.0
    for epoch in range(cfg.train.epochs):
        lr_now        = optimizer.param_groups[0]["lr"]
        loss          = train_one_epoch(model, train_loader, optimizer, cfg, device, amp_ctx, train_ds.transform)
        train_metrics = evaluate(model, train_eval_loader, cfg, device, amp_ctx, train_ds.transform)
        metrics       = evaluate(model, val_loader, cfg, device, amp_ctx, val_ds.transform)
        scheduler.step()

        c_index = metrics.get("c_index", float("nan"))
        score   = c_index if not math.isnan(c_index) else -1.0
        print(
            f"  [fold {fold}] Epoch {epoch+1:3d} | lr={lr_now:.2e} | loss={loss:.4f} | "
            f"train_c_index={train_metrics['c_index']:.4f} | "
            f"val_c_index={metrics['c_index']:.4f}"
        )

        if WANDB_AVAILABLE:
            wandb.log({
                "train/loss":              loss,
                "train/lr":                lr_now,
                "train/c_index":           train_metrics["c_index"],
                "val_performance/c_index": metrics["c_index"],
            }, step=epoch + 1)

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch":            epoch + 1,
                    "val_c_index":      best_score,
                    "fold":             fold,
                },
                ckpt_path,
            )

    # 이 fold의 최고 체크포인트로 val 예측을 다시 계산 — fold 간 pooling(OOF c-index)에 사용
    if best_score > -1.0:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    oof = evaluate(model, val_loader, cfg, device, amp_ctx, val_ds.transform)

    print(f"  → fold {fold} best val c-index: {best_score:.4f} (checkpoint: {ckpt_path.name})")

    if WANDB_AVAILABLE:
        wandb.run.summary["best_val_c_index"] = best_score
        wandb.finish()

    return best_score, oof


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

    mode = "precomputed features" if cfg.data.precomputed else "raw image (--image)"
    print(f"Mode: {mode}")
    if args.fusion:
        K = int(cluster_centroids.shape[0])
        print(f"Model: LateFusionViT (ViT+ABMIL + ClusterHistogram, K={K})")
    else:
        print("Model: PatchViT (ViT+ABMIL baseline)")
    print(f"Dataset: {args.dataset}  |  {cfg.data.n_folds}-fold stratified CV")
    print(
        f"AMP=bfloat16 | batch={cfg.train.cox_batch_size} patients (Cox risk set 단위) "
        f"| cnn_chunk={cfg.train.cnn_chunk_size} | workers={cfg.data.num_workers}"
    )

    ckpt_dir = Path(__file__).parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prefix = "F_" if args.fusion else "N_"
    wandb_group = prefix + datetime.now().strftime("%m%d::%H%M")

    fold_scores = []
    oof_risks, oof_times, oof_events = [], [], []
    for fold in range(cfg.data.n_folds):
        print(f"\n=== Fold {fold+1}/{cfg.data.n_folds} ===")
        score, oof = _run_fold(fold, cfg, args, cluster_centroids, device, amp_ctx, ckpt_dir, wandb_group)
        fold_scores.append(score)
        oof_risks.append(oof["risks"])
        oof_times.append(oof["times"])
        oof_events.append(oof["events"])

    fold_scores = np.array(fold_scores)
    pooled = compute_survival_metrics(
        np.concatenate(oof_risks), np.concatenate(oof_times), np.concatenate(oof_events),
    )
    pooled_c_index = pooled["c_index"]

    print("\n=== Cross-Validation Summary ===")
    print(f"Per-fold val c-index : {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Mean ± std           : {fold_scores.mean():.4f} ± {fold_scores.std():.4f}")
    print(f"Pooled OOF c-index   : {pooled_c_index:.4f}  (전체 {len(np.concatenate(oof_risks))}명 pooling)")

    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    send_slack(
        f":white_check_mark: *Path-ViT ({args.dataset.upper()} OS) {cfg.data.n_folds}-fold CV 완료*\n"
        f"> Pooled OOF C-index: *{pooled_c_index:.4f}* | fold 평균: {fold_scores.mean():.4f} ± {fold_scores.std():.4f}\n"
        f"> 소요 시간: {h}h {m}m {s}s"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        load_env()
        send_slack(f":x: *Path-ViT (OS) 학습 에러*\n```{type(e).__name__}: {e}```")
        raise
