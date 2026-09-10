"""TCGA-BRCA M1(WSI only)/M2(WSI+Clinical) internal+external 학습/평가 — train.py --M1/--M2의
BRCA ClusterPool 버전. --clinical 유무로 M1/M2 두 슬롯을 전부 이 스크립트 하나로 커버한다.
train_brca_m4.py와 동일 골격(RNA 브랜치만 없음) — BRCASlideDataset이 요구하는 rna_df는 모델이
안 쓰지만 그대로 채워 넣는다(PAAD M1/M2와 동일 관례).

사용법:
    python -m scripts.train_brca_m1_m2 --seed 84 --fold 0 --n-folds 5              # M1
    python -m scripts.train_brca_m1_m2 --seed 84 --clinical --clinical-staging --clinical-lr-mult 100 --fold 0 --n-folds 5  # M2
"""
import argparse
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from models.vit_m1 import ViT_M1
from models.vit_m2 import ViT_M2
from models.clinical_encoder import age_stats_from_csv, stage_stats_from_csv
from train import set_seed, _build_scheduler, _log_line, train_one_epoch, evaluate, WANDB_AVAILABLE, _branch_param_groups
from utils.losses import fit_survival_bins
from utils.metrics import compute_time_dependent_auc
from scripts.brca_common import (
    CLINICAL_PATH, BRCASlideDataset, _identity_collate, load_case_table, load_case_table_kfold,
    load_rna_matrix, MANIFEST_PATH, resolve_external_tss,
)

if WANDB_AVAILABLE:
    import wandb


def _make_amp_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clinical", action="store_true", help="켜면 M2(WSI+Clinical), 안 켜면 M1(WSI only).")
    parser.add_argument("--clinical-staging", action="store_true")
    parser.add_argument("--clinical-lr-mult", type=float, default=1.0)
    parser.add_argument("--patch-keep-frac", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--group-ts", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--cluster-centroids-path", type=str, default="data/cluster_centroids_brca_uni.pt",
        help="data/fit_clusters_brca_uni.py 산출물 경로 — 항상 ClusterPool을 쓴다(2026-09 확정 "
             "레시피, M1/M2도 예외 없음).",
    )
    parser.add_argument("--surv-loss", type=str, default="cox", choices=["cox", "nll_surv", "both"])
    parser.add_argument("--nll-n-bins", type=int, default=4)
    parser.add_argument("--nll-cox-weight", type=float, default=1.0)
    parser.add_argument("--external-tss", type=str, default="multi")
    parser.add_argument("--rna-dummy-genes", type=int, default=10,
                         help="BRCASlideDataset이 요구하는 rna_df 채우기용 — 모델(ViT_M1/ViT_M2)은 "
                              "RNA 브랜치가 없어 실제로 안 쓴다(PAAD M1/M2와 동일 관례).")
    args = parser.parse_args()
    external_tss, ext_tag = resolve_external_tss(args.external_tss)

    cfg = Config()
    cfg.data.seed = cfg.train.seed = args.seed
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx(device)
    start_time = datetime.now()

    dummy_gene_ids = pd.read_csv("data/brca_rna_gene_selection_consistency/selected_genes.csv")["gene_id"].tolist()[: args.rna_dummy_genes]
    stage_stats = stage_stats_from_csv(CLINICAL_PATH) if (args.clinical and args.clinical_staging) else None

    if args.fold is not None:
        cases = load_case_table_kfold(args.seed, args.fold, args.n_folds, external_tss=external_tss)
    else:
        cases = load_case_table(args.seed, external_tss=external_tss)
    rna_df = load_rna_matrix(dummy_gene_ids)
    manifest = pd.read_csv(MANIFEST_PATH)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATH)
    print(f"case 수: {len(cases)}  (train={int((cases['split']=='train').sum())}, "
          f"val={int((cases['split']=='val').sum())}, test={int((cases['split']=='test').sum())}, "
          f"external={int((cases['split']=='external').sum())} [tss={external_tss}])")

    if args.clinical:
        model = ViT_M2(
            cfg.model, age_mean=age_mean, age_std=age_std, precomputed=True, backbone="uni",
            use_staging=args.clinical_staging, stage_stats=stage_stats,
            cluster_pool=True, cluster_centroids_path=args.cluster_centroids_path,
            surv_n_classes=(args.nll_n_bins if args.surv_loss in ("nll_surv", "both") else 1),
        ).to(device)
        model_prefix = "BRCA_M2"
        if args.clinical_staging:
            model_prefix += "_STG"
        if args.clinical_lr_mult != 1.0:
            model_prefix += f"_CLR{args.clinical_lr_mult:g}"
    else:
        model = ViT_M1(
            cfg.model, precomputed=True, backbone="uni",
            cluster_pool=True, cluster_centroids_path=args.cluster_centroids_path,
            surv_n_classes=(args.nll_n_bins if args.surv_loss in ("nll_surv", "both") else 1),
        ).to(device)
        model_prefix = "BRCA_M1"
    model_prefix += "_CLUSTERPOOL"
    if args.surv_loss in ("nll_surv", "both"):
        model_prefix += f"_NLLSURV{args.nll_n_bins}"
    if args.surv_loss == "both":
        model_prefix += f"_NLLCOX{args.nll_cox_weight:g}"
    fold_suffix = f"_fold{args.fold}of{args.n_folds}" if args.fold is not None else ""

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    train_ds     = BRCASlideDataset(cases[cases["split"] == "train"],    rna_df, manifest)
    val_ds       = BRCASlideDataset(cases[cases["split"] == "val"],      rna_df, manifest)
    test_ds      = BRCASlideDataset(cases[cases["split"] == "test"],     rna_df, manifest)
    external_ds  = BRCASlideDataset(cases[cases["split"] == "external"], rna_df, manifest) if external_tss else None
    train_loader      = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **dl_kwargs)
    val_loader        = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader       = DataLoader(test_ds,  shuffle=False, **dl_kwargs)
    external_loader   = DataLoader(external_ds, shuffle=False, **dl_kwargs) if external_ds is not None else None

    print(f"Model: {model_prefix} ({type(model).__name__}) | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"lr={cfg.train.lr:.1e} | weight_decay={cfg.train.weight_decay:.1e} | epochs={cfg.train.epochs} | "
          f"patch_keep_frac={args.patch_keep_frac} | cox_batch_size={cfg.train.cox_batch_size}")

    run_ts = datetime.now().strftime("%m%d::%H%M")
    group_ts = args.group_ts or run_ts
    wandb_group = f"{model_prefix}_{group_ts}"
    if WANDB_AVAILABLE:
        wandb.init(
            project="Path-ViT", name=f"BRCA_{model_prefix}_seed{cfg.train.seed}{fold_suffix}_{run_ts}",
            group=wandb_group,
            config={"epochs": cfg.train.epochs, "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay,
                    "seed": cfg.train.seed, "patch_keep_frac": args.patch_keep_frac,
                    "clinical_lr_mult": args.clinical_lr_mult, "fold": args.fold, "n_folds": args.n_folds,
                    "model": model_prefix, "dataset": "brca"},
        )

    if args.clinical and args.clinical_lr_mult != 1.0:
        groups = _branch_param_groups(model)
        param_groups = []
        if groups["clinical"]:
            param_groups.append({"params": groups["clinical"], "lr": cfg.train.lr * args.clinical_lr_mult})
        if groups["other"]:
            param_groups.append({"params": groups["other"], "lr": cfg.train.lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
        print(f"branch-lr-mult 적용: clinical={args.clinical_lr_mult}x({len(groups['clinical'])}개 텐서), "
              f"other=1x({len(groups['other'])}개 텐서)")
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        )
    scheduler = _build_scheduler(optimizer, cfg)

    ckpt_dir = Path(__file__).parent.parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"survival_brca_best_{model_prefix.lower()}{ext_tag.lower()}_seed{args.seed}{fold_suffix}.pt"

    nll_bin_edges = None
    if args.surv_loss in ("nll_surv", "both"):
        _train_labels = cases[cases["split"] == "train"]
        nll_bin_edges = fit_survival_bins(
            _train_labels["OS_time"].to_numpy(), _train_labels["OS_event"].to_numpy(), n_bins=args.nll_n_bins,
        )
        print(f"[nll_surv] train split {len(_train_labels)}명 기준 시간-구간 경계({args.nll_n_bins}bins): {nll_bin_edges}")

    best_score, best_metrics = -1.0, {}
    for epoch in range(cfg.train.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(
            model, train_loader, optimizer, cfg, device, amp_ctx, None,
            patch_keep_frac=args.patch_keep_frac, rna_aux_weight=0.0,
            surv_loss=args.surv_loss, nll_bin_edges=nll_bin_edges, nll_cox_weight=args.nll_cox_weight,
        )
        train_metrics = evaluate(model, train_eval_loader, cfg, device, amp_ctx, None)
        metrics = evaluate(model, val_loader, cfg, device, amp_ctx, None)
        val_td_auc = compute_time_dependent_auc(
            train_metrics["times"], train_metrics["events"], metrics["times"], metrics["events"], metrics["risks"],
        )
        scheduler.step()
        c_index = metrics.get("c_index", float("nan"))
        score = c_index if c_index == c_index else -1.0
        print(f"Epoch {epoch+1:3d} | lr={lr_now:.2e} | loss={loss:.4f} | "
              f"train_c_index={train_metrics['c_index']:.4f} | " + _log_line("val", metrics, val_td_auc))
        if WANDB_AVAILABLE:
            wandb.log({"train/loss": loss, "train/lr": lr_now, "train/c_index": train_metrics["c_index"],
                       "val_performance/c_index": metrics["c_index"], "val_performance/hr": metrics["hr"],
                       "val_performance/log_rank_p": metrics["log_rank_p"],
                       "val_performance/auc_mean": val_td_auc["auc_mean"]}, step=epoch + 1)
        if score > best_score:
            best_score = score
            best_metrics = {**metrics, "epoch": epoch + 1}
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_c_index": best_score}, ckpt_path)
            print(f"  -> checkpoint saved (c_index={best_score:.4f})")
            if WANDB_AVAILABLE:
                wandb.run.summary["best_val_c_index"] = best_score
                wandb.run.summary["best_epoch"] = epoch + 1

    final_train_metrics = evaluate(model, train_eval_loader, cfg, device, amp_ctx, None)
    final_test_metrics = evaluate(model, test_loader, cfg, device, amp_ctx, None)
    print(f"\n=== BRCA Internal Test (마지막 epoch {epoch + 1} 모델, best-val 선택 없음) ===")
    print(_log_line("final_test", final_test_metrics))
    import csv as _csv
    fe_pred_dir = Path(__file__).parent.parent / ".logs" / "kfold_preds"
    fe_pred_dir.mkdir(parents=True, exist_ok=True)
    fe_pred_path = fe_pred_dir / f"brca_{model_prefix}{ext_tag}_FINALEPOCH_seed{args.seed}{fold_suffix}.csv"
    with open(fe_pred_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
        for cid, risk, t, e in zip(final_test_metrics["case_ids"], final_test_metrics["risks"],
                                    final_test_metrics["times"], final_test_metrics["events"]):
            writer.writerow([cid, risk, t, e])
    print(f"  -> final-epoch predictions saved: {fe_pred_path}")
    if external_loader is not None:
        final_external_metrics = evaluate(model, external_loader, cfg, device, amp_ctx, None)
        fe_ext_pred_dir = Path(__file__).parent.parent / ".logs" / "external_preds"
        fe_ext_pred_dir.mkdir(parents=True, exist_ok=True)
        fe_ext_pred_path = fe_ext_pred_dir / f"brca_{model_prefix}{ext_tag}_FINALEPOCH_seed{args.seed}{fold_suffix}.csv"
        with open(fe_ext_pred_path, "w", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
            for cid, risk, t, e in zip(final_external_metrics["case_ids"], final_external_metrics["risks"],
                                        final_external_metrics["times"], final_external_metrics["events"]):
                writer.writerow([cid, risk, t, e])
        print(f"  -> final-epoch external predictions saved: {fe_ext_pred_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    train_metrics_final = evaluate(model, train_eval_loader, cfg, device, amp_ctx, None)
    test_metrics = evaluate(model, test_loader, cfg, device, amp_ctx, None)
    test_td_auc = compute_time_dependent_auc(
        train_metrics_final["times"], train_metrics_final["events"],
        test_metrics["times"], test_metrics["events"], test_metrics["risks"],
    )
    print(f"\n=== BRCA Internal Test (best checkpoint epoch {ckpt['epoch']}) ===")
    print(_log_line("test", test_metrics, test_td_auc))
    import csv
    pred_dir = Path(__file__).parent.parent / ".logs" / "kfold_preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"brca_{model_prefix}{ext_tag}_seed{args.seed}{fold_suffix}.csv"
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
        for cid, risk, t, e in zip(test_metrics["case_ids"], test_metrics["risks"],
                                    test_metrics["times"], test_metrics["events"]):
            writer.writerow([cid, risk, t, e])
    print(f"  -> internal predictions saved: {pred_path}")

    if external_loader is not None:
        external_metrics = evaluate(model, external_loader, cfg, device, amp_ctx, None)
        external_td_auc = compute_time_dependent_auc(
            train_metrics_final["times"], train_metrics_final["events"],
            external_metrics["times"], external_metrics["events"], external_metrics["risks"],
        )
        print(f"\n=== BRCA External Test (institution={external_tss}, best checkpoint) ===")
        print(_log_line("external", external_metrics, external_td_auc))
        pred_dir = Path(__file__).parent.parent / ".logs" / "external_preds"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / f"brca_{model_prefix}{ext_tag}_seed{args.seed}{fold_suffix}.csv"
        with open(pred_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
            for cid, risk, t, e in zip(external_metrics["case_ids"], external_metrics["risks"],
                                        external_metrics["times"], external_metrics["events"]):
                writer.writerow([cid, risk, t, e])
        print(f"  -> external predictions saved: {pred_path}")

    if WANDB_AVAILABLE:
        wandb.finish()
    elapsed = datetime.now() - start_time
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    print(f"\n소요 시간: {h}h {m}m {s}s")


if __name__ == "__main__":
    main()
