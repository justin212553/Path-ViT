"""
HDP_Cluster 학습 스크립트 — models/hdp_cluster.py::HDPCluster. train_light.py(--HDP, patch
forward 없음)와 달리 이 모델은 슬라이드별 patch 공간 배치(occupancy map -> CNN)와 patch
단위(feature+군집 soft weight -> MLP) 계산이 필요해 별도 스크립트로 뗐다 — train.py급 무거운
CNN/ViT 인코더는 없고(uni2native feature는 이미 사전 추출돼 있음), coords 기반 map 구성 +
작은 CNN/MLP forward만 있어 train.py보다는 훨씬 가볍다.

[배경] 2026-09-01 — HDP(비지도 k-means 군집, 결정론적 4*K차원 통계, 학습 파라미터 0개)가
여러 정제를 거쳐도 M7과 통계적으로 계속 동률이라, 원래 계획에서 "리스크가 크다"고 미뤄뒀던
GrowthPatternCNN(침윤전선/성장 패턴)과 MaturityMLP(성숙도)를 다시 넣어 152개 생존 라벨로
end-to-end 학습시켜본다(사용자 결정).

WSI patch 데이터(coords/features)는 WSISurvivalDataset(feature_backbone="uni2native")이
그대로 제공한다 — train.py/M1~M7이 HPC에서 이미 이 backbone으로 검증한 정식 경로라, 이 프로젝트가
직접 h5/pt 파일을 읽는 커스텀 로더보다 훨씬 안전하다(2026-09-01, 사용자 지적으로 커스텀
WSILookup을 걷어내고 이걸로 교체). train_light.py류가 patient dict의 "coords"/"features" key를
그냥 안 읽었을 뿐, WSISurvivalDataset 자체는 항상 이 값을 채워서 반환한다.

⚠️ 로컬 환경엔 uni2native 변환 트리(data/patches_{tcga,cptac}/tiles/*/features_uni2native.pt)가
없어(원본 h5만 있음) 이 스크립트를 로컬에서 끝까지 실행 검증은 못 했다 — HPC(M1~M7이 이미 이
backbone으로 학습된 환경)에서 첫 실행 결과를 확인해야 한다.

사용법:
    python train_hdp_cluster.py --dataset tcga --seed 84 --external --fold 0 --n-folds 5
"""
import argparse
import csv
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids_intersection
from models.hdp_cluster import HDPCluster
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df, STAGE_FIELDS
from utils import load_env
from utils.losses import cox_ph_loss
from utils.metrics import compute_survival_metrics

from train_light import _load_cluster_histograms, ClusterHistLookup  # HDP 40차원 통계 재사용

_ROOT = Path(__file__).resolve().parent
CENTROIDS_PATH = _ROOT / "data" / "cluster_centroids_uni2native.pt"
GRID_STRIDE = 512  # data/fit_clusters_uni2native.py 확인 시 coords 간격(level-0 px)과 동일


def _soft_weights(dist: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """data/compute_cluster_features_uni2native.py::_soft_weights와 동일 공식(torch 버전,
    GPU에서 바로 계산). dist: (N, K)."""
    mu = dist.mean(dim=1, keepdim=True)
    sigma = dist.std(dim=1, keepdim=True)
    z = (dist - mu) / (sigma + eps)
    z = z - z.min(dim=1, keepdim=True).values
    w = torch.exp(-z)
    return w / w.sum(dim=1, keepdim=True)


def _build_occupancy_map(coords: torch.Tensor, weights: torch.Tensor, k: int) -> torch.Tensor:
    """coords(N,2, level-0 px, GRID_STRIDE 간격 정규 격자) + weights(N,K) -> (1,K,H,W) 맵.
    각 patch는 정확히 격자 한 칸에 대응(겹침 없음) — scatter만 하면 되고 보간 불필요."""
    coords = coords.float()
    gx = torch.round((coords[:, 0] - coords[:, 0].min()) / GRID_STRIDE).long()
    gy = torch.round((coords[:, 1] - coords[:, 1].min()) / GRID_STRIDE).long()
    h, w = int(gy.max().item()) + 1, int(gx.max().item()) + 1
    grid = torch.zeros(k, h, w, device=weights.device, dtype=weights.dtype)
    grid[:, gy, gx] = weights.T
    return grid.unsqueeze(0)  # (1, K, H, W)


class PrecomputeCache:
    """case_id -> (occupancy_map별 리스트, feat_cat, w_cat). soft weight/맵 구성은 고정된
    centroid에 대한 결정론적 계산이라(학습 파라미터 무관) 환자당 1회만 계산해 캐싱한다 —
    CNN/MLP의 학습 가능한 파라미터는 매번 새로 forward(캐싱 대상 아님), 그 앞단(거리/soft
    weight/맵 구성)만 캐싱해 epoch 반복 비용을 줄인다."""

    def __init__(self, centroids: torch.Tensor):
        self.centroids = centroids
        self.k = centroids.shape[0]
        self._cache: dict[str, tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]] = {}

    def __call__(self, case_id: str, patient_slides: list[dict], device) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        if case_id in self._cache:
            maps, feat_cat, w_cat = self._cache[case_id]
            return maps, feat_cat.to(device), w_cat.to(device)

        maps, all_feat, all_w = [], [], []
        for slide in patient_slides:
            coords = slide["coords"].to(device)
            feat = slide["features"].to(device).float()
            dist = torch.linalg.norm(feat.unsqueeze(1) - self.centroids.unsqueeze(0), dim=-1)  # (N, K)
            w = _soft_weights(dist)
            maps.append(_build_occupancy_map(coords, w, self.k))
            all_feat.append(feat)
            all_w.append(w)
        feat_cat = torch.cat(all_feat, dim=0)
        w_cat = torch.cat(all_w, dim=0)
        self._cache[case_id] = ([m.cpu() for m in maps], feat_cat.cpu(), w_cat.cpu())
        return maps, feat_cat, w_cat


def _identity_collate(batch: list) -> list:
    return batch[0]


def _patient_risk(model: HDPCluster, patient_slides, precompute: PrecomputeCache,
                   cluster_hist_lookup: ClusterHistLookup, device) -> torch.Tensor:
    p = patient_slides[0]
    case_id = p["case_id"]
    maps, feat_cat, w_cat = precompute(case_id, patient_slides, device)

    growth_vecs = [model.growth_cnn(m.to(device)) for m in maps]
    growth_vec = torch.stack(growth_vecs).mean(dim=0)
    maturity_scalar = model.maturity_mlp(feat_cat, w_cat)

    cluster_hist = cluster_hist_lookup[case_id].to(device, non_blocking=True)
    margin_kwargs = {}
    if getattr(model, "use_margin", False):
        margin_kwargs["margin_ord"] = p["margin_ord"].to(device, non_blocking=True)
    if getattr(model, "use_staging", False):
        margin_kwargs["stage_ord"] = {f: p[f].to(device, non_blocking=True) for f in STAGE_FIELDS}

    return model(
        p["age_years"].to(device, non_blocking=True),
        p["sex_idx"].to(device, non_blocking=True),
        p["rna"].to(device, non_blocking=True),
        cluster_hist,
        growth_vec,
        maturity_scalar,
        **margin_kwargs,
    )


def train_one_epoch(model, loader, optimizer, device, batch_size, precompute, cluster_hist_lookup) -> float:
    model.train()
    total_loss, total_batches = 0.0, 0
    risks, times, events = [], [], []

    def _flush():
        nonlocal risks, times, events, total_loss, total_batches
        if not risks:
            return
        loss = cox_ph_loss(torch.cat(risks), torch.cat(times).to(device), torch.cat(events).to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
        risks.clear(); times.clear(); events.clear()

    for patient_slides in loader:
        if len(patient_slides) == 0:
            continue
        risks.append(_patient_risk(model, patient_slides, precompute, cluster_hist_lookup, device))
        times.append(patient_slides[0]["OS_time"])
        events.append(patient_slides[0]["OS_event"])
        if len(risks) >= batch_size:
            _flush()
    _flush()
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, precompute, cluster_hist_lookup) -> dict:
    model.eval()
    all_risks, all_times, all_events, all_case_ids = [], [], [], []
    for patient_slides in loader:
        if len(patient_slides) == 0:
            continue
        risk = _patient_risk(model, patient_slides, precompute, cluster_hist_lookup, device)
        all_risks.append(risk.float().item())
        all_times.append(float(patient_slides[0]["OS_time"].item()))
        all_events.append(int(patient_slides[0]["OS_event"].item()))
        all_case_ids.append(patient_slides[0]["case_id"])
    risks, times, events = np.array(all_risks), np.array(all_times), np.array(all_events)
    return {**compute_survival_metrics(risks, times, events),
            "risks": risks, "times": times, "events": events, "case_ids": all_case_ids}


def _build_scheduler(optimizer, epochs, warmup):
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    # 2026-09-01: train_hdp_pretrain_cluster.py의 lr sweep(같은 GrowthPatternCNN+MaturityMLP
    # 아키텍처, paper/hdp/*.log) 결과를 여기도 그대로 적용 — 1e-3에서 3e-5로 기본값 변경.
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--cox-batch-size", type=int, default=16)
    parser.add_argument("--growth-dim", type=int, default=8)
    parser.add_argument("--group-ts", type=str, default=None)
    parser.add_argument("--eval-external-ckpt", type=str, default=None)
    args = parser.parse_args()

    load_env()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config()
    cfg.data.seed = args.seed
    external_dataset = ("cptac" if args.dataset == "tcga" else "tcga") if args.external else None

    print("[준비] centroids/cluster feature 로드")
    centroids = torch.load(CENTROIDS_PATH, map_location=device, weights_only=False).float()
    k = centroids.shape[0]
    hist_datasets = [args.dataset] + ([external_dataset] if external_dataset else [])
    precompute = PrecomputeCache(centroids)
    cluster_hist_lookup, hist_dim = _load_cluster_histograms(hist_datasets, source="cluster")
    print(f"  cluster feature 차원={hist_dim}, K={k}")

    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset]))
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)

    model_prefix = f"HDP_CLUSTER_INT1500_STG_R_GROWTH{args.growth_dim}"
    if args.lr != 3e-5:
        # train_light.py의 _LR{lr:.0e} 관례와 동일 — lr sweep(같은 seed/fold)에서 checkpoint/
        # kfold_preds 파일명이 서로 덮어써지지 않게 한다.
        model_prefix += f"_LR{args.lr:.0e}"
    if args.fold is not None:
        model_prefix += f"_FOLD{args.fold}OF{args.n_folds}"

    model = HDPCluster(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(rna_gene_ids),
        hist_dim=hist_dim, k=k, feat_dim=centroids.shape[1], growth_dim=args.growth_dim,
        use_margin=True, margin_stats=margin_stats, use_age_sex=True,
        use_staging=True, stage_stats=stage_stats,
    ).to(device)
    print(f"Model: {model_prefix} | params={sum(p.numel() for p in model.parameters()):,}")

    ds_kwargs = dict(with_clinical=True, with_margin=True, with_staging=True,
                      with_rna=True, rna_gene_ids=rna_gene_ids, feature_backbone="uni2native")
    split_kwargs = dict(fold=args.fold, n_folds=args.n_folds)
    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)

    if args.eval_external_ckpt:
        if not external_dataset:
            raise ValueError("--eval-external-ckpt는 --external과 함께 써야 합니다.")
        ckpt = torch.load(args.eval_external_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        external_ds = WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", **ds_kwargs)
        external_loader = DataLoader(external_ds, shuffle=False, **dl_kwargs)
        metrics = evaluate(model, external_loader, device, precompute, cluster_hist_lookup)
        print(f"  external c_index={metrics['c_index']:.4f} | HR={metrics['hr']:.3f} | "
              f"log_rank_p={metrics['log_rank_p']:.4f}")
        pred_dir = _ROOT / ".logs" / "external_preds"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / f"{external_dataset}_{model_prefix}_seed{args.seed}_fold{args.fold}of{args.n_folds}.csv"
        with open(pred_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
            for cid, r, t, e in zip(metrics["case_ids"], metrics["risks"], metrics["times"], metrics["events"]):
                writer.writerow([cid, r, t, e])
        print(f"  -> saved: {pred_path}")
        return

    train_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="train", **ds_kwargs, **split_kwargs)
    val_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="val", **ds_kwargs, **split_kwargs)
    test_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="test", **ds_kwargs, **split_kwargs)
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)
    print(f"Dataset: {args.dataset} fold{args.fold}/{args.n_folds} | "
          f"Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")

    run_ts = datetime.now().strftime("%m%d::%H%M")
    if WANDB_AVAILABLE:
        wandb.init(project="Path-ViT", name=f"{args.dataset.upper()}_{model_prefix}_seed{args.seed}_{run_ts}",
                    group=f"{model_prefix}_{args.group_ts or run_ts}",
                    config={"epochs": args.epochs, "lr": args.lr, "seed": args.seed, "model": model_prefix})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _build_scheduler(optimizer, args.epochs, warmup=max(1, args.epochs // 10))
    ckpt_dir = _ROOT / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"survival_{args.dataset}_best_{model_prefix.lower()}_seed{args.seed}_light.pt"

    best_score, epochs_since_improvement = -1.0, 0
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, args.cox_batch_size,
                                precompute, cluster_hist_lookup)
        val_metrics = evaluate(model, val_loader, device, precompute, cluster_hist_lookup)
        scheduler.step()
        print(f"Epoch {epoch:3d} | loss={loss:.4f} | val_c_index={val_metrics['c_index']:.4f} | "
              f"val_HR={val_metrics['hr']:.3f} | val_logrank_p={val_metrics['log_rank_p']:.4f}")
        if WANDB_AVAILABLE:
            wandb.log({"train/loss": loss, "val/c_index": val_metrics["c_index"],
                       "val/hr": val_metrics["hr"], "val/log_rank_p": val_metrics["log_rank_p"]})
        score = val_metrics["c_index"]
        if not np.isnan(score) and score > best_score:
            best_score = score
            epochs_since_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_c_index": score}, ckpt_path)
            print(f"  -> checkpoint saved (c_index={score:.4f})")
        else:
            epochs_since_improvement += 1
        if args.patience and epochs_since_improvement >= args.patience:
            print(f"  -> early stopping (patience={args.patience})")
            break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, precompute, cluster_hist_lookup)
    print(f"\n=== Internal Test (best epoch {ckpt['epoch']}) ===")
    print(f"test_c_index={test_metrics['c_index']:.4f} | HR={test_metrics['hr']:.3f} | "
          f"log_rank_p={test_metrics['log_rank_p']:.4f}")

    pred_dir = _ROOT / ".logs" / "kfold_preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{args.dataset}_{model_prefix}_seed{args.seed}_fold{args.fold}of{args.n_folds}.csv"
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
        for cid, r, t, e in zip(test_metrics["case_ids"], test_metrics["risks"],
                                  test_metrics["times"], test_metrics["events"]):
            writer.writerow([cid, r, t, e])
    print(f"  -> internal fold predictions saved: {pred_path}")

    if external_dataset:
        external_ds = WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", **ds_kwargs)
        external_loader = DataLoader(external_ds, shuffle=False, **dl_kwargs)
        ext_metrics = evaluate(model, external_loader, device, precompute, cluster_hist_lookup)
        print(f"\n=== External Test ({external_dataset}) ===")
        print(f"external_c_index={ext_metrics['c_index']:.4f} | HR={ext_metrics['hr']:.3f} | "
              f"log_rank_p={ext_metrics['log_rank_p']:.4f}")


if __name__ == "__main__":
    main()
