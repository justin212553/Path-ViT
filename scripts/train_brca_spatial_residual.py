"""
Stage2 — 공간정보 잔차 branch(models/spatial_residual.py::SpatialResidualBranch)를 TCGA-BRCA로
재현. scripts/train_spatial_residual.py(PAAD)의 BRCA 버전 — "PAAD(152명)이 너무 작아서
공간 잔차 branch가 신호를 못 찾는 거 아니냐"는 가설을, WSI가 이미 M7 대비 순증분 기여를
보여준 BRCA(1058명, PAAD의 약 7배, findings_backlog.md 최상위 발견)로 직접 검증한다.

Stage1은 scripts/train_brca_m4.py가 만든 기존 체크포인트(BRCA_PMA_TOP1500_SS_AUX, seed42,
RNA는 고분산 상위 1500개)를 그대로 재사용 — 재학습 불필요. BRCA는 UNI backbone 사전추출
feature(features_uni.pt)만 쓰므로(raw 이미지 모드 자체가 없음) precompute가 매우 빠르다.

사용 (PathViT-ray 환경):
  python -m scripts.train_brca_spatial_residual --layer-type gcn
"""
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import Config
from models.vit_pma import ViT_PMA
from models.rna_predictor import RNAPredictionHead
from models.clinical_encoder import age_stats_from_csv
from models.spatial_residual import SpatialResidualBranch, shuffle_margin_loss
from scripts.brca_common import CLINICAL_PATH, BRCASlideDataset, load_case_table, load_rna_matrix
from utils import load_env
from utils.losses import cox_ph_loss
from utils.metrics import compute_survival_metrics

DEFAULT_STAGE1_CKPT = "models/checkpoint/survival_brca_best_brca_pma_top1500_ss_aux_seed42.pt"
GENE_LIST_PATH = Path("data/brca_rna_gene_selection/selected_genes_top_1500.csv")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", type=str, default=DEFAULT_STAGE1_CKPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--edge-dropout", type=float, default=0.2)
    p.add_argument("--layer-type", type=str, default="gcn", choices=["gcn", "attention"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--shuffle-margin", type=float, default=0.3)
    p.add_argument("--shuffle-margin-weight", type=float, default=0.3)
    p.add_argument("--group-ts", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def _precompute_patient(model, patient_slides, device):
    """scripts/train_spatial_residual.py::_precompute_patient의 BRCA 버전 — BRCASlideDataset은
    항상 features(UNI 사전추출)만 반환하므로 raw 이미지 분기가 아예 필요 없다.
    """
    rna = patient_slides[0]["rna"].to(device, non_blocking=True)
    z_rna = model.encode_rna(rna)

    slide_embeds, slides_cache = [], []
    for slide in patient_slides:
        coords = slide["coords"].to(device, non_blocking=True)
        out = model(coords, features=slide["features"], rna_context=z_rna)
        slide_embeds.append(out["embed"])
        slides_cache.append((out["patch_tokens"].float().cpu(), coords.cpu()))

    patient_embed = torch.stack(slide_embeds).mean(dim=0)
    age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
    sex_idx   = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
    combined  = model.combine_with_clinical_rna(patient_embed, age_years, sex_idx, z_rna)
    base_risk = model.risk_head(combined.unsqueeze(0)).view(()).float().cpu()

    return {
        "base_risk": base_risk,
        "slides":    slides_cache,
        "OS_time":   patient_slides[0]["OS_time"],
        "OS_event":  patient_slides[0]["OS_event"],
    }


def _precompute_split(model, ds, device, desc: str) -> list[dict]:
    cache = [_precompute_patient(model, ds[i], device) for i in range(len(ds))]
    print(f"  [{desc}] {len(cache)}명 캐싱 완료")
    return cache


def _branch_risk(branch: SpatialResidualBranch, patient: dict, device) -> torch.Tensor:
    slide_reprs = [
        branch.encode(pt.to(device, non_blocking=True), c.to(device, non_blocking=True))
        for pt, c in patient["slides"]
    ]
    pooled = torch.stack(slide_reprs).mean(dim=0)
    return branch.head(branch.head_drop(pooled)).view(())


@torch.no_grad()
def _evaluate(branch: SpatialResidualBranch, cache: list[dict], device) -> dict:
    branch.eval()
    risks, times, events = [], [], []
    for patient in cache:
        final_risk = patient["base_risk"].to(device) + _branch_risk(branch, patient, device)
        risks.append(final_risk.item())
        times.append(float(patient["OS_time"].item()))
        events.append(int(patient["OS_event"].item()))
    return compute_survival_metrics(np.array(risks), np.array(times), np.array(events))


def main():
    load_env()
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = Path("data/spatial_residual_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    stage1_tag = Path(args.stage1_ckpt).stem
    cache_path = cache_dir / f"brca_seed{args.seed}_{stage1_tag}.pt"

    if cache_path.exists():
        print(f"Stage1 precompute 캐시 로드: {cache_path}")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        train_cache, val_cache, test_cache = cached["train"], cached["val"], cached["test"]
        ckpt = {"epoch": cached["stage1_epoch"], "val_c_index": cached["stage1_val_c_index"]}
    else:
        cfg = Config()
        gene_ids = pd.read_csv(GENE_LIST_PATH)["gene_id"].tolist()
        rna_input_dim = len(gene_ids)
        cases = load_case_table(args.seed)
        rna_df = load_rna_matrix(gene_ids)
        manifest = pd.read_csv("data/brca_slide_manifest.csv")
        age_mean, age_std = age_stats_from_csv(CLINICAL_PATH)

        train_ds = BRCASlideDataset(cases[cases["split"] == "train"], rna_df, manifest)
        val_ds   = BRCASlideDataset(cases[cases["split"] == "val"],   rna_df, manifest)
        test_ds  = BRCASlideDataset(cases[cases["split"] == "test"],  rna_df, manifest)
        print(f"case 수: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

        stage1 = ViT_PMA(
            cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
            precomputed=True, backbone="uni",
        ).to(device)
        stage1.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, rna_input_dim).to(device)
        ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
        stage1.load_state_dict(ckpt["model_state_dict"])
        stage1.eval()
        for p in stage1.parameters():
            p.requires_grad_(False)
        print(f"Stage1 로드 완료: {args.stage1_ckpt} (epoch={ckpt.get('epoch')}, "
              f"val_c_index={ckpt.get('val_c_index')})")

        print("Stage1 forward로 base_risk + patch_tokens 사전계산 중(1회성)...")
        train_cache = _precompute_split(stage1, train_ds, device, "train")
        val_cache   = _precompute_split(stage1, val_ds,   device, "val")
        test_cache  = _precompute_split(stage1, test_ds,  device, "test")
        del stage1
        torch.cuda.empty_cache()

        torch.save({
            "train": train_cache, "val": val_cache, "test": test_cache,
            "stage1_epoch": ckpt.get("epoch"), "stage1_val_c_index": ckpt.get("val_c_index"),
        }, cache_path)
        print(f"Stage1 precompute 캐시 저장: {cache_path}")

    patch_dim = train_cache[0]["slides"][0][0].shape[-1]
    branch = SpatialResidualBranch(
        patch_dim=patch_dim, hidden_dim=args.hidden_dim, k=args.k,
        num_layers=args.num_layers, dropout=args.dropout, edge_dropout=args.edge_dropout,
        layer_type=args.layer_type,
    ).to(device)
    optimizer = torch.optim.AdamW(branch.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_ts = datetime.now().strftime("%m%d::%H%M")
    group_ts = args.group_ts or run_ts
    global WANDB_AVAILABLE
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project="Path-ViT", name=f"BRCA_SPATIALRESID_{args.layer_type}_seed{args.seed}_{run_ts}",
                group=f"BRCA_SPATIALRESID_{group_ts}",
                config={**vars(args), "stage1_val_c_index": ckpt.get("val_c_index")},
                settings=wandb.Settings(init_timeout=120),
            )
        except Exception as e:
            print(f"wandb.init 실패({e!r}) — 이 run은 wandb 없이 진행합니다.")
            WANDB_AVAILABLE = False

    ckpt_dir = Path("models/checkpoint")
    ckpt_dir.mkdir(exist_ok=True)
    branch_ckpt_path = ckpt_dir / f"spatial_residual_brca_{args.layer_type}_seed{args.seed}_best.pt"

    best_val_c = -1.0
    epochs_since_improve = 0
    for epoch in range(args.epochs):
        branch.train()
        random.shuffle(train_cache)
        risks, times, events, margin_losses = [], [], [], []
        total_loss, total_steps = 0.0, 0
        batch_size = 16

        def _flush():
            nonlocal risks, times, events, margin_losses, total_loss, total_steps
            if not risks:
                return
            risk_t  = torch.stack(risks)
            time_t  = torch.cat(times).to(device)
            event_t = torch.cat(events).to(device)
            loss = cox_ph_loss(risk_t, time_t, event_t)
            if margin_losses:
                loss = loss + args.shuffle_margin_weight * torch.stack(margin_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_steps += 1
            risks, times, events, margin_losses = [], [], [], []

        for patient in train_cache:
            final_risk = patient["base_risk"].to(device) + _branch_risk(branch, patient, device)
            risks.append(final_risk.view(1))
            times.append(patient["OS_time"])
            events.append(patient["OS_event"])
            pt, c = patient["slides"][0]
            margin_losses.append(
                shuffle_margin_loss(branch, pt.to(device), c.to(device), margin=args.shuffle_margin)
            )
            if len(risks) >= batch_size:
                _flush()
        _flush()

        val_metrics = _evaluate(branch, val_cache, device)
        loss_avg = total_loss / max(total_steps, 1)
        print(
            f"Epoch {epoch+1:3d} | loss={loss_avg:.4f} | "
            f"val_c_index={val_metrics['c_index']:.4f} | val_HR={val_metrics['hr']:.3f} | "
            f"val_logrank_p={val_metrics['log_rank_p']:.4f}"
        )
        if WANDB_AVAILABLE:
            wandb.log({
                "train/loss": loss_avg,
                "val_performance/c_index": val_metrics["c_index"],
                "val_performance/hr": val_metrics["hr"],
                "val_performance/log_rank_p": val_metrics["log_rank_p"],
            }, step=epoch + 1)

        if val_metrics["c_index"] > best_val_c:
            best_val_c = val_metrics["c_index"]
            epochs_since_improve = 0
            torch.save({"branch_state_dict": branch.state_dict(), "epoch": epoch + 1,
                        "val_c_index": best_val_c}, branch_ckpt_path)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"  -> val_c_index가 {args.patience} epoch 동안 갱신되지 않아 조기 종료 "
                      f"(epoch {epoch+1}, best={best_val_c:.4f} @ epoch {epoch+1-epochs_since_improve})")
                break

    best = torch.load(branch_ckpt_path, map_location=device, weights_only=False)
    branch.load_state_dict(best["branch_state_dict"])
    test_metrics = _evaluate(branch, test_cache, device)
    print(f"\n=== BRCA Internal Test 성능 (best checkpoint, epoch {best['epoch']}) ===")
    print(f"test_c_index={test_metrics['c_index']:.4f} | test_HR={test_metrics['hr']:.3f} "
          f"| test_logrank_p={test_metrics['log_rank_p']:.4f}")
    if WANDB_AVAILABLE:
        wandb.run.summary["test_c_index"] = test_metrics["c_index"]
        wandb.run.summary["test_hr"] = test_metrics["hr"]
        wandb.run.summary["test_log_rank_p"] = test_metrics["log_rank_p"]
        wandb.finish()


if __name__ == "__main__":
    main()
