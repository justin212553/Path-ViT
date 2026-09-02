"""
HDP_Cluster 학습 스크립트 — models/hdp_cluster.py::HDPCluster. train_light.py(--HDP, patch
forward 없음)와 달리 이 모델은 슬라이드별 patch 공간 배치(occupancy map -> CNN)와 patch
단위(feature+군집 soft weight -> MLP) 계산이 필요해 별도 스크립트로 뗐다 — train.py급 무거운
CNN/ViT 인코더는 없고(uni2native feature는 이미 h5에 추출돼 있음), coords 기반 map 구성 +
작은 CNN/MLP forward만 있어 train.py보다는 훨씬 가볍다.

[배경] 2026-09-01 — HDP(비지도 k-means 군집, 결정론적 4*K차원 통계, 학습 파라미터 0개)가
여러 정제를 거쳐도 M7과 통계적으로 계속 동률이라, 원래 계획에서 "리스크가 크다"고 미뤄뒀던
GrowthPatternCNN(침윤전선/성장 패턴)과 MaturityMLP(성숙도)를 다시 넣어 152개 생존 라벨로
end-to-end 학습시켜본다(사용자 결정).

RNA/clinical/OS_time/OS_event는 WSISurvivalDataset(with_wsi 요청 안 함, train_light.py와 동일
관례)에서 그대로 가져오고, WSI 쪽(uni2native feature+coords)은 data/uni2h_official_features/
*.h5에서 case_id로 직접 lookup한다(train_light.py의 --HDP cluster_hist lookup과 같은 패턴을
patch 단위로 확장한 것).

사용법:
    python train_hdp_cluster.py --dataset tcga --seed 84 --external --fold 0 --n-folds 5
"""
import argparse
import math
import random
from datetime import datetime
from pathlib import Path

import h5py
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
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids_intersection
from models.hdp_cluster import HDPCluster
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df, STAGE_FIELDS
from utils import load_env
from utils.losses import cox_ph_loss
from utils.metrics import compute_survival_metrics

from train_light import _load_cluster_histograms, ClusterHistLookup  # HDP 40차원 통계 재사용

_ROOT = Path(__file__).resolve().parent
FEATURES_ROOT = _ROOT / "data" / "uni2h_official_features"
CENTROIDS_PATH = _ROOT / "data" / "cluster_centroids_uni2native.pt"
CASE_ID_TOKENS = {"tcga": 3, "cptac": 2}
GRID_STRIDE = 512  # data/fit_clusters_uni2native.py 확인 시 coords 간격(level-0 px)과 동일


def _case_id_from_stem(stem: str, dataset: str) -> str:
    return "-".join(stem.split("-")[: CASE_ID_TOKENS[dataset]])


def _build_slide_index(datasets: list[str]) -> dict[str, list[Path]]:
    """case_id -> 그 환자의 uni2native h5 파일 경로 리스트(슬라이드 여러 장 가능)."""
    index: dict[str, list[Path]] = {}
    for ds in datasets:
        for h5_path in sorted((FEATURES_ROOT / ds).glob("*.h5")):
            cid = _case_id_from_stem(h5_path.stem, ds)
            index.setdefault(cid, []).append(h5_path)
    return index


def _soft_weights(dist: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """data/compute_cluster_features_uni2native.py::_soft_weights와 동일 공식."""
    mu = dist.mean(axis=1, keepdims=True)
    sigma = dist.std(axis=1, keepdims=True)
    z = (dist - mu) / (sigma + eps)
    z = z - z.min(axis=1, keepdims=True)
    w = np.exp(-z)
    return w / w.sum(axis=1, keepdims=True)


def _build_occupancy_map(coords: np.ndarray, weights: np.ndarray, k: int) -> torch.Tensor:
    """coords(N,2, level-0 px, GRID_STRIDE 간격 정규 격자) + weights(N,K) -> (1,K,H,W) 맵.
    각 patch는 정확히 격자 한 칸에 대응(겹침 없음) — scatter만 하면 되고 보간 불필요."""
    gx = np.round((coords[:, 0] - coords[:, 0].min()) / GRID_STRIDE).astype(int)
    gy = np.round((coords[:, 1] - coords[:, 1].min()) / GRID_STRIDE).astype(int)
    h, w = gy.max() + 1, gx.max() + 1
    grid = np.zeros((k, h, w), dtype=np.float32)
    grid[:, gy, gx] = weights.T.astype(np.float32)
    return torch.from_numpy(grid).unsqueeze(0)  # (1, K, H, W)


class WSILookup:
    """case_id -> 슬라이드별 (coords, features, soft_weights) 리스트. soft weight는 고정된
    centroid에 대한 결정론적 계산이라(학습 중 안 바뀜) 환자당 1회만 계산해 in-memory 캐싱한다
    — 안 하면 epoch마다(--epochs 100) 같은 h5 I/O+거리 계산을 반복해 학습이 크게 느려진다
    (2026-09-01 스모크 테스트에서 캐싱 없이 2epoch에 ~5분 확인 후 추가). 전체 코호트(~350명)가
    한 번에 캐싱돼도 수 GB 수준이라 메모리 문제 없음."""

    def __init__(self, slide_index: dict[str, list[Path]], centroids: np.ndarray):
        self.slide_index = slide_index
        self.centroids = centroids
        self.k = centroids.shape[0]
        self._cache: dict[str, list[dict]] = {}

    def __call__(self, case_id: str) -> list[dict] | None:
        if case_id in self._cache:
            return self._cache[case_id]
        paths = self.slide_index.get(case_id)
        if not paths:
            return None
        out = []
        for p in paths:
            with h5py.File(p, "r") as f:
                feat = f["features"][0].astype(np.float32)
                coords = f["coords"][0]
            dist = np.linalg.norm(feat[:, None, :] - self.centroids[None, :, :], axis=-1)
            w = _soft_weights(dist)
            out.append({"coords": coords, "features": feat, "weights": w})
        self._cache[case_id] = out
        return out


def _identity_collate(batch: list) -> list:
    return batch[0]


def _patient_risk(model: HDPCluster, patient_slides, wsi_lookup: WSILookup,
                   cluster_hist_lookup: ClusterHistLookup, device) -> torch.Tensor:
    p = patient_slides[0]
    case_id = p["case_id"]
    slides = wsi_lookup(case_id)

    if slides:
        growth_vecs = []
        all_feat, all_w = [], []
        for s in slides:
            occ = _build_occupancy_map(s["coords"], s["weights"], wsi_lookup.k).to(device)
            growth_vecs.append(model.growth_cnn(occ))
            all_feat.append(torch.from_numpy(s["features"]).to(device))
            all_w.append(torch.from_numpy(s["weights"].astype(np.float32)).to(device))
        growth_vec = torch.stack(growth_vecs).mean(dim=0)
        feat_cat = torch.cat(all_feat, dim=0)
        w_cat = torch.cat(all_w, dim=0)
        maturity_scalar = model.maturity_mlp(feat_cat, w_cat)
    else:
        growth_vec = torch.zeros(model.growth_cnn.proj.out_features, device=device)
        maturity_scalar = torch.zeros((), device=device)

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


def train_one_epoch(model, loader, optimizer, device, batch_size, wsi_lookup, cluster_hist_lookup) -> float:
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
        risks.append(_patient_risk(model, patient_slides, wsi_lookup, cluster_hist_lookup, device))
        times.append(patient_slides[0]["OS_time"])
        events.append(patient_slides[0]["OS_event"])
        if len(risks) >= batch_size:
            _flush()
    _flush()
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, wsi_lookup, cluster_hist_lookup) -> dict:
    model.eval()
    all_risks, all_times, all_events, all_case_ids = [], [], [], []
    for patient_slides in loader:
        if len(patient_slides) == 0:
            continue
        risk = _patient_risk(model, patient_slides, wsi_lookup, cluster_hist_lookup, device)
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

    print("[준비] centroids/slide index/cluster feature 로드")
    centroids = torch.load(CENTROIDS_PATH, weights_only=False).numpy().astype(np.float32)
    k = centroids.shape[0]
    hist_datasets = [args.dataset] + ([external_dataset] if external_dataset else [])
    slide_index = _build_slide_index(hist_datasets)
    wsi_lookup = WSILookup(slide_index, centroids)
    cluster_hist_lookup, hist_dim = _load_cluster_histograms(hist_datasets, source="cluster")
    print(f"  slide_index: {len(slide_index)}명, cluster feature 차원={hist_dim}")

    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.dataset])
    import pandas as pd
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset]))
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)

    model_prefix = f"HDP_CLUSTER_INT1500_STG_R_GROWTH{args.growth_dim}"
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
                      with_rna=True, rna_gene_ids=rna_gene_ids)
    split_kwargs = dict(fold=args.fold, n_folds=args.n_folds)
    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)

    if args.eval_external_ckpt:
        if not external_dataset:
            raise ValueError("--eval-external-ckpt는 --external과 함께 써야 합니다.")
        ckpt = torch.load(args.eval_external_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        external_ds = WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", **ds_kwargs)
        external_loader = DataLoader(external_ds, shuffle=False, **dl_kwargs)
        metrics = evaluate(model, external_loader, device, wsi_lookup, cluster_hist_lookup)
        print(f"  external c_index={metrics['c_index']:.4f} | HR={metrics['hr']:.3f} | "
              f"log_rank_p={metrics['log_rank_p']:.4f}")
        import csv
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
                                wsi_lookup, cluster_hist_lookup)
        val_metrics = evaluate(model, val_loader, device, wsi_lookup, cluster_hist_lookup)
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
    test_metrics = evaluate(model, test_loader, device, wsi_lookup, cluster_hist_lookup)
    print(f"\n=== Internal Test (best epoch {ckpt['epoch']}) ===")
    print(f"test_c_index={test_metrics['c_index']:.4f} | HR={test_metrics['hr']:.3f} | "
          f"log_rank_p={test_metrics['log_rank_p']:.4f}")

    import csv
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
        ext_metrics = evaluate(model, external_loader, device, wsi_lookup, cluster_hist_lookup)
        print(f"\n=== External Test ({external_dataset}) ===")
        print(f"external_c_index={ext_metrics['c_index']:.4f} | HR={ext_metrics['hr']:.3f} | "
              f"log_rank_p={ext_metrics['log_rank_p']:.4f}")


if __name__ == "__main__":
    main()
