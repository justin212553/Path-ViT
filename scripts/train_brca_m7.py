"""
TCGA-BRCA M7(ClinicalRNAOnly, WSI 없음) internal 학습/평가 — train_light.py --M7의 BRCA 버전.

M4(scripts/train_brca_m4.py, ViT_PMA)와 "같은 환경"에서 비교하기 위한 대조군이다(사용자
지시: "같은 환경일 때 M7을 넘냐 안 넘냐가 문제"). 반드시 지켜야 하는 동일 조건:
  - case 목록/6:2:2 split: scripts/brca_common.py (M4와 동일 --seed)
  - RNA 유전자셋: scripts/select_brca_rna_genes.py가 뽑은 고분산 상위 1500개
    (PDAC 전용 literature_1500 대신 — scripts/extract_brca_rna.py 참조)

모델 자체(models/clinical_rna_only.py::ClinicalRNAOnly)와 학습 루프(_patient_risk/
train_one_epoch/evaluate/_build_scheduler)는 train_light.py 실제 코드를 그대로 import해서
쓴다 — 새로 재구현하면 로직이 미묘하게 어긋날 위험이 있어(reference_repro_m4.py/m7.py가
레퍼런스 코드를 직접 import하는 것과 같은 원칙), 우리 자신의 검증된 코드를 그대로 재사용한다.
레퍼런스 M7 레시피(epochs=100, patience=20)를 기본값으로 쓴다(models/clinical_rna_only.py
docstring 참조).

사용법:
    python -m scripts.train_brca_m7 --seed 42
    python -m scripts.train_brca_m7 --seed 42 --epochs 100 --patience 20 --n-genes 1500
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
from models.clinical_rna_only import ClinicalRNAOnly
from models.clinical_encoder import age_stats_from_csv
from train_light import (
    set_seed, _build_scheduler, _log_line, train_one_epoch, evaluate, WANDB_AVAILABLE,
)
from utils.metrics import compute_time_dependent_auc
from scripts.brca_common import (
    CLINICAL_PATH, BRCACaseDataset, _identity_collate, load_case_table, load_rna_matrix,
)

if WANDB_AVAILABLE:
    import wandb

OUT_DIR = Path("data/brca_rna_gene_selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-genes", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=100, help="레퍼런스 M7 레시피 기본값.")
    parser.add_argument("--patience", type=int, default=20, help="레퍼런스 M7 레시피 기본값.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--group-ts", type=str, default=None)
    args = parser.parse_args()

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

    gene_path = OUT_DIR / f"selected_genes_top_{args.n_genes}.csv"
    if not gene_path.exists():
        raise FileNotFoundError(
            f"{gene_path} 없음 — 먼저 실행: python -m scripts.select_brca_rna_genes "
            f"--seed {args.seed} --n-genes {args.n_genes}"
        )
    import pandas as pd
    gene_ids = pd.read_csv(gene_path)["gene_id"].tolist()
    rna_input_dim = len(gene_ids)

    cases = load_case_table(args.seed)
    rna_df = load_rna_matrix(gene_ids)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATH)
    print(f"case 수: {len(cases)}  (train={int((cases['split']=='train').sum())}, "
          f"val={int((cases['split']=='val').sum())}, test={int((cases['split']=='test').sum())})")
    print(f"RNA 유전자 수: {rna_input_dim} (top{args.n_genes}, 고분산 기준, seed={args.seed})")
    print(f"age_mean={age_mean:.2f} age_std={age_std:.2f} (전체 코호트 기준, train.py 관례와 동일)")

    model = ClinicalRNAOnly(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim).to(device)
    model_prefix = f"BRCA_M7_TOP{args.n_genes}"

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    train_ds = BRCACaseDataset(cases[cases["split"] == "train"], rna_df)
    val_ds   = BRCACaseDataset(cases[cases["split"] == "val"],   rna_df)
    test_ds  = BRCACaseDataset(cases[cases["split"] == "test"],  rna_df)
    train_loader      = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **dl_kwargs)
    val_loader        = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader       = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    print(f"Model: {model_prefix} ({type(model).__name__}) | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"lr={cfg.light.lr:.1e} | weight_decay={cfg.light.weight_decay:.1e} | "
          f"epochs={cfg.light.epochs} | patience={args.patience} | cox_batch_size={cfg.light.cox_batch_size}")

    run_ts = datetime.now().strftime("%m%d::%H%M")
    group_ts = args.group_ts or run_ts
    wandb_group = f"{model_prefix}_{group_ts}"
    if WANDB_AVAILABLE:
        wandb.init(
            project="Path-ViT",
            name=f"BRCA_{model_prefix}_seed{cfg.light.seed}_{run_ts}",
            group=wandb_group,
            config={
                "epochs": cfg.light.epochs, "lr": cfg.light.lr, "weight_decay": cfg.light.weight_decay,
                "seed": cfg.light.seed, "patience": args.patience, "n_genes": args.n_genes,
                "rna_input_dim": rna_input_dim, "model": model_prefix, "dataset": "brca",
            },
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.light.lr, weight_decay=cfg.light.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg)

    ckpt_dir = Path(__file__).parent.parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"survival_brca_best_{model_prefix.lower()}_seed{args.seed}.pt"

    best_score, best_metrics, epochs_since_improvement = -1.0, {}, 0
    for epoch in range(cfg.light.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(model, train_loader, optimizer, device, cfg.light.cox_batch_size)
        train_metrics = evaluate(model, train_eval_loader, device)
        metrics = evaluate(model, val_loader, device)
        val_td_auc = compute_time_dependent_auc(
            train_metrics["times"], train_metrics["events"], metrics["times"], metrics["events"], metrics["risks"],
        )
        scheduler.step()

        c_index = metrics.get("c_index", float("nan"))
        score = c_index if c_index == c_index else -1.0  # NaN != NaN
        print(f"Epoch {epoch+1:3d} | lr={lr_now:.2e} | loss={loss:.4f} | "
              f"train_c_index={train_metrics['c_index']:.4f} | " + _log_line("val", metrics, val_td_auc))

        if WANDB_AVAILABLE:
            wandb.log({
                "train/loss": loss, "train/lr": lr_now, "train/c_index": train_metrics["c_index"],
                "val_performance/c_index": metrics["c_index"], "val_performance/hr": metrics["hr"],
                "val_performance/log_rank_p": metrics["log_rank_p"],
                "val_performance/auc_mean": val_td_auc["auc_mean"],
            }, step=epoch + 1)

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
    if WANDB_AVAILABLE:
        wandb.run.summary["test_c_index"] = test_metrics["c_index"]
        wandb.run.summary["test_hr"] = test_metrics["hr"]
        wandb.run.summary["test_log_rank_p"] = test_metrics["log_rank_p"]
        wandb.run.summary["test_auc_mean"] = test_td_auc["auc_mean"]
        wandb.finish()

    elapsed = datetime.now() - start_time
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    print(f"\n소요 시간: {h}h {m}m {s}s")


if __name__ == "__main__":
    main()
