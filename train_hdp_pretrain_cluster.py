"""
HDP_Pretrain_Cluster 학습 스크립트 — models/hdp_cluster.py::HDPCluster를 K=1(진짜 종양 함량
스칼라 하나, PanNuke로 학습된 models/tumor_content_head.py::TumorContentHead 출력)로 재사용한다.
train_hdp_cluster.py(K=10 비지도 군집)와 구조는 완전히 같고 "K개 채널이 뭐냐"만 다르다 —
GrowthPatternCNN/MaturityMLP 둘 다 k를 파라미터로 받게 설계돼 있어 새 모델 클래스가 필요 없다.

[배경] 2026-09-01 — HDP_Pretrain(mean/heterogeneity/dispersion/frac_high 4개 결정론적
요약 통계, 학습 파라미터 0개)이 M7과 여전히 동률(internal -0.0128, external +0.0040, 둘 다
노이즈 수준). "정보를 너무 압축한 거 아니냐"는 질문(사용자) — 4개 요약 통계 대신 전체 공간
map을 CNN이 직접 보게 해서, 압축이 병목이었는지 직접 검증한다. HDP_Cluster(K=10 군집 버전)의
CNN+MLP 추가가 이미 M7 대비 무변화였던 전례가 있어(internal +0.0013, external +0.0005 — 순수
군집 버전 대비도 무변화), 여기서도 비슷한 결과가 나올 가능성이 높지만, "진짜 라벨 기반 신호"
에 대해서도 압축 여부가 무관한지는 직접 확인해야 한다.

WSI patch 데이터(coords/features)는 train_hdp_cluster.py와 동일하게 WSISurvivalDataset
(feature_backbone="uni2native")에서 가져온다. ⚠️ 로컬에서 끝까지 검증 못 함(train_hdp_cluster.py와
동일한 이유 — uni2native 변환 트리가 로컬에 없음), HPC에서 확인 필요.

사용법:
    python train_hdp_pretrain_cluster.py --dataset tcga --seed 84 --external --fold 0 --n-folds 5
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
from models.tumor_content_head import TumorContentHead
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df, STAGE_FIELDS
from utils import load_env
from utils.losses import cox_ph_loss
from utils.metrics import compute_survival_metrics

from train_light import _load_cluster_histograms, ClusterHistLookup  # HDP_Pretrain 4차원 통계 재사용

_ROOT = Path(__file__).resolve().parent
# models/checkpoint/는 git-ignore 대상이라(2026-09-01 HPC에서 FileNotFoundError로 확인) data/
# 밑에 별도로 복사해둔 걸 쓴다 — git pull만으로 HPC에도 따라오게.
HEAD_PATH = _ROOT / "data" / "hdp_pretrain_tumor_content_head.pt"
GRID_STRIDE = 512  # data/fit_clusters_uni2native.py 확인 시 coords 간격(level-0 px)과 동일


def _load_head(device) -> TumorContentHead:
    ckpt = torch.load(HEAD_PATH, map_location=device, weights_only=False)
    head = TumorContentHead(in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"]).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    head.requires_grad_(False)
    return head


def _build_occupancy_map(coords: torch.Tensor, weights: torch.Tensor, k: int) -> torch.Tensor:
    """train_hdp_cluster.py와 동일 — coords(N,2, level-0 px, GRID_STRIDE 정규 격자) +
    weights(N,K) -> (1,K,H,W) 맵. patch 하나 = 격자 한 칸(겹침 없음, 보간 불필요)."""
    coords = coords.float()
    gx = torch.round((coords[:, 0] - coords[:, 0].min()) / GRID_STRIDE).long()
    gy = torch.round((coords[:, 1] - coords[:, 1].min()) / GRID_STRIDE).long()
    h, w = int(gy.max().item()) + 1, int(gx.max().item()) + 1
    grid = torch.zeros(k, h, w, device=weights.device, dtype=weights.dtype)
    grid[:, gy, gx] = weights.T
    return grid.unsqueeze(0)  # (1, K, H, W)


class PrecomputeCache:
    """case_id -> (occupancy_map별 리스트, feat_cat, w_cat). train_hdp_cluster.py와 같은
    캐싱 목적이지만 "weight"가 군집까지 거리-softmax가 아니라 frozen TumorContentHead의
    출력(N,1) 그 자체다 — K=1이라 softmax/정규화가 필요 없다(이미 sigmoid로 0~1)."""

    def __init__(self, head: TumorContentHead):
        self.head = head
        self.k = 1
        self._cache: dict[str, tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]] = {}

    def __call__(self, case_id: str, patient_slides: list[dict], device) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        if case_id in self._cache:
            maps, feat_cat, w_cat = self._cache[case_id]
            return maps, feat_cat.to(device), w_cat.to(device)

        maps, all_feat, all_w = [], [], []
        with torch.no_grad():
            for slide in patient_slides:
                coords = slide["coords"].to(device)
                feat = slide["features"].to(device).float()
                w = self.head(feat).unsqueeze(-1)  # (N, 1) — 이미 sigmoid로 0~1
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
    parser.add_argument("--lr", type=float, default=1e-3)
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

    print("[준비] pretrain head/pretrain 4차원 통계 로드")
    head = _load_head(device)
    feat_dim = head.net[1].in_features if hasattr(head.net[1], "in_features") else 1536
    precompute = PrecomputeCache(head)
    hist_datasets = [args.dataset] + ([external_dataset] if external_dataset else [])
    cluster_hist_lookup, hist_dim = _load_cluster_histograms(hist_datasets, source="pretrain")
    print(f"  pretrain feature 차원={hist_dim}, K=1(진짜 종양 함량 스칼라)")

    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset]))
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)

    model_prefix = f"HDP_PRETRAIN_CLUSTER_INT1500_STG_R_GROWTH{args.growth_dim}"
    if args.fold is not None:
        model_prefix += f"_FOLD{args.fold}OF{args.n_folds}"

    model = HDPCluster(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(rna_gene_ids),
        hist_dim=hist_dim, k=1, feat_dim=feat_dim, growth_dim=args.growth_dim,
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
