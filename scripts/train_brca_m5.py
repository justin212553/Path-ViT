"""TCGA-BRCA M5(ClinicalOnly, WSI/RNA 없음) internal+external 학습/평가 — train_light.py --M5의
BRCA 버전. train_brca_m6.py와 동일 골격, clinical만 남기고 RNA를 뺀 버전.

사용법:
    python -m scripts.train_brca_m5 --seed 84 --clinical-staging --fold 0 --n-folds 5
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from models.clinical_only import ClinicalOnly
from models.clinical_encoder import age_stats_from_csv, stage_stats_from_csv
from train_light import set_seed, _build_scheduler, _log_line, train_one_epoch, evaluate, WANDB_AVAILABLE
from utils.losses import fit_survival_bins
from utils.metrics import compute_time_dependent_auc
from scripts.brca_common import (
    CLINICAL_PATH, BRCACaseDataset, _identity_collate, load_case_table, load_case_table_kfold,
    load_rna_matrix, resolve_external_tss,
)

if WANDB_AVAILABLE:
    import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clinical-staging", action="store_true")
    parser.add_argument("--raw-linear", action="store_true", default=True,
                         help="PAAD 최종 관례(2026-08-21 확정) — clinical 브랜치는 예외 없이 "
                              "raw feature 직결(ClinicalEncoder MLP 없음). 기본 True.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--group-ts", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--surv-loss", type=str, default="cox", choices=["cox", "nll_surv", "both"])
    parser.add_argument("--nll-n-bins", type=int, default=4)
    parser.add_argument("--nll-cox-weight", type=float, default=1.0)
    parser.add_argument("--external-tss", type=str, default="multi")
    # 아무 RNA 유전자셋(모델엔 안 씀, BRCACaseDataset이 dict에 rna 필드를 무조건 채워서 그대로
    # 재사용 — M6와 동일한 이유, PAAD의 M1/M2가 아무 클러스터 센트로이드나 요구 안 하는 것과 대칭).
    parser.add_argument("--rna-dummy-genes", type=int, default=10)
    args = parser.parse_args()
    external_tss, ext_tag = resolve_external_tss(args.external_tss)

    cfg = Config()
    cfg.data.seed = cfg.light.seed = args.seed
    cfg.light.epochs = args.epochs
    if args.lr is not None:
        cfg.light.lr = args.lr
    if args.weight_decay is not None:
        cfg.light.weight_decay = args.weight_decay
    set_seed(cfg.light.seed)
    device = torch.device(cfg.light.device if torch.cuda.is_available() else "cpu")
    start_time = datetime.now()

    import pandas as pd
    gene_path = Path("data/brca_rna_gene_selection_consistency/selected_genes.csv")
    dummy_gene_ids = pd.read_csv(gene_path)["gene_id"].tolist()[: args.rna_dummy_genes]

    stage_stats = stage_stats_from_csv(CLINICAL_PATH) if args.clinical_staging else None
    if args.fold is not None:
        cases = load_case_table_kfold(args.seed, args.fold, args.n_folds, external_tss=external_tss)
    else:
        cases = load_case_table(args.seed, external_tss=external_tss)
    rna_df = load_rna_matrix(dummy_gene_ids)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATH)
    print(f"case 수: {len(cases)}  (train={int((cases['split']=='train').sum())}, "
          f"val={int((cases['split']=='val').sum())}, test={int((cases['split']=='test').sum())}, "
          f"external={int((cases['split']=='external').sum())} [tss={external_tss}])")

    model = ClinicalOnly(
        cfg.model, age_mean=age_mean, age_std=age_std,
        use_staging=args.clinical_staging, stage_stats=stage_stats, raw_linear=args.raw_linear,
        surv_n_classes=(args.nll_n_bins if args.surv_loss in ("nll_surv", "both") else 1),
    ).to(device)
    model_prefix = "BRCA_M5"
    if args.clinical_staging:
        model_prefix += "_STG"
    if args.surv_loss in ("nll_surv", "both"):
        model_prefix += f"_NLLSURV{args.nll_n_bins}"
    if args.surv_loss == "both":
        model_prefix += f"_NLLCOX{args.nll_cox_weight:g}"
    fold_suffix = f"_fold{args.fold}of{args.n_folds}" if args.fold is not None else ""

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    stg = args.clinical_staging
    train_ds     = BRCACaseDataset(cases[cases["split"] == "train"],    rna_df, with_staging=stg)
    val_ds       = BRCACaseDataset(cases[cases["split"] == "val"],      rna_df, with_staging=stg)
    test_ds      = BRCACaseDataset(cases[cases["split"] == "test"],     rna_df, with_staging=stg)
    external_ds  = BRCACaseDataset(cases[cases["split"] == "external"], rna_df, with_staging=stg) if external_tss else None
    train_loader      = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **dl_kwargs)
    val_loader        = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader       = DataLoader(test_ds,  shuffle=False, **dl_kwargs)
    external_loader   = DataLoader(external_ds, shuffle=False, **dl_kwargs) if external_ds is not None else None

    print(f"Model: {model_prefix} ({type(model).__name__}) | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"lr={cfg.light.lr:.1e} | weight_decay={cfg.light.weight_decay:.1e} | "
          f"epochs={cfg.light.epochs} | patience={args.patience}")

    run_ts = datetime.now().strftime("%m%d::%H%M")
    group_ts = args.group_ts or run_ts
    wandb_group = f"{model_prefix}_{group_ts}"
    if WANDB_AVAILABLE:
        wandb.init(
            project="Path-ViT", name=f"BRCA_{model_prefix}_seed{cfg.light.seed}{fold_suffix}_{run_ts}",
            group=wandb_group,
            config={"epochs": cfg.light.epochs, "lr": cfg.light.lr, "weight_decay": cfg.light.weight_decay,
                    "seed": cfg.light.seed, "fold": args.fold, "n_folds": args.n_folds,
                    "model": model_prefix, "dataset": "brca"},
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.light.lr, weight_decay=cfg.light.weight_decay)
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

    best_score, best_metrics, epochs_since_improvement = -1.0, {}, 0
    for epoch in range(cfg.light.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(
            model, train_loader, optimizer, device, cfg.light.cox_batch_size,
            surv_loss=args.surv_loss, nll_bin_edges=nll_bin_edges, nll_cox_weight=args.nll_cox_weight,
        )
        train_metrics = evaluate(model, train_eval_loader, device)
        metrics = evaluate(model, val_loader, device)
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
            epochs_since_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_c_index": best_score}, ckpt_path)
            print(f"  -> checkpoint saved (c_index={best_score:.4f})")
            if WANDB_AVAILABLE:
                wandb.run.summary["best_val_c_index"] = best_score
                wandb.run.summary["best_epoch"] = epoch + 1
        else:
            epochs_since_improvement += 1
            if args.patience is not None and epochs_since_improvement >= args.patience:
                print(f"  -> early stopping (patience={args.patience}, "
                      f"best epoch {best_metrics.get('epoch', '-')} c_index={best_score:.4f})")
                break

    final_test_metrics = evaluate(model, test_loader, device)
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
    if WANDB_AVAILABLE:
        wandb.run.summary["final_epoch_test_c_index"] = final_test_metrics["c_index"]
    if external_loader is not None:
        final_external_metrics = evaluate(model, external_loader, device)
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
    train_metrics_final = evaluate(model, train_eval_loader, device)
    test_metrics = evaluate(model, test_loader, device)
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
    if WANDB_AVAILABLE:
        wandb.run.summary["test_c_index"] = test_metrics["c_index"]

    if external_loader is not None:
        external_metrics = evaluate(model, external_loader, device)
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
            wandb.run.summary["external_c_index"] = external_metrics["c_index"]

    if WANDB_AVAILABLE:
        wandb.finish()
    elapsed = datetime.now() - start_time
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    print(f"\n소요 시간: {h}h {m}m {s}s")


if __name__ == "__main__":
    main()
