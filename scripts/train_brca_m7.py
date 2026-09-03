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
from models.clinical_encoder import age_stats_from_csv, stage_stats_from_csv
from train_light import (
    set_seed, _build_scheduler, _log_line, train_one_epoch, evaluate, WANDB_AVAILABLE,
)
from utils.metrics import compute_time_dependent_auc
from scripts.brca_common import (
    CLINICAL_PATH, BRCACaseDataset, _identity_collate, load_case_table, load_case_table_kfold,
    load_rna_matrix, EXTERNAL_TSS,
)

if WANDB_AVAILABLE:
    import wandb

OUT_DIR = Path("data/brca_rna_gene_selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-genes", type=int, default=1500)
    parser.add_argument("--gene-selection", type=str, default="variance", choices=["variance", "cox"],
                         help="2026-09-02: 'variance'(기본, scripts/select_brca_rna_genes.py의 "
                              "고분산 상위 N개, data/brca_rna_gene_selection/) 또는 "
                              "'cox'(scripts/select_brca_rna_genes_cox.py의 생존 라벨 기반 "
                              "univariate Cox score test 상위 N개, "
                              "data/brca_rna_gene_selection_cox/) 중 선택.")
    parser.add_argument("--clinical-staging", action="store_true",
                         help="2026-09-02: ClinicalEncoder 입력에 AJCC 병기(ajcc_t/n/m, "
                              "tumor_grade)를 추가한다(train.py --clinical-staging과 동일 관례). "
                              "BRCA는 tumor_grade가 GDC에 항상 결측이라 그 필드는 always "
                              "known_flag=0으로 안전하게 무시된다(scripts/extract_brca_labels.py "
                              "참조). margin(residual_disease)은 BRCA 전체 0/1098 결측이라 지원 안 함.")
    parser.add_argument("--epochs", type=int, default=100, help="레퍼런스 M7 레시피 기본값.")
    parser.add_argument("--patience", type=int, default=20, help="레퍼런스 M7 레시피 기본값.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--group-ts", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None,
                         help="2026-09-01: 주어지면(0-based) 기존 단일 6:2:2 대신 PAAD와 동일한 "
                              "K-fold(scripts/brca_common.py::load_case_table_kfold)를 쓴다 — "
                              "M4(scripts/train_brca_m4.py)와 반드시 동일 --seed/--fold/--n-folds로 "
                              "비교해야 같은 데이터 분할이 된다.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--external-tss", type=str, default=EXTERNAL_TSS,
        help=f"institution-level external holdout(TCGA barcode 2번째 세그먼트, 기본 "
             f"{EXTERNAL_TSS!r}). 'none'이면 external 없이 기존 동작. M4와 반드시 동일 값을 "
             "써야 비교가 성립한다(scripts/brca_common.py 참조).",
    )
    args = parser.parse_args()
    external_tss = None if args.external_tss.lower() == "none" else args.external_tss
    ext_tag = f"_EXTTSS{external_tss}" if external_tss else ""  # None이면 파일명에 접미사 없음

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

    gene_dir = OUT_DIR if args.gene_selection == "variance" else Path("data/brca_rna_gene_selection_cox")
    gene_path = gene_dir / f"selected_genes_top_{args.n_genes}.csv"
    if not gene_path.exists():
        select_module = "select_brca_rna_genes" if args.gene_selection == "variance" else "select_brca_rna_genes_cox"
        raise FileNotFoundError(
            f"{gene_path} 없음 — 먼저 실행: python -m scripts.{select_module} "
            f"--seed {args.seed} --n-genes {args.n_genes}"
        )
    import pandas as pd
    gene_ids = pd.read_csv(gene_path)["gene_id"].tolist()
    rna_input_dim = len(gene_ids)
    stage_stats = stage_stats_from_csv(CLINICAL_PATH) if args.clinical_staging else None

    if args.fold is not None:
        cases = load_case_table_kfold(args.seed, args.fold, args.n_folds, external_tss=external_tss)
    else:
        cases = load_case_table(args.seed, external_tss=external_tss)
    rna_df = load_rna_matrix(gene_ids)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATH)
    print(f"case 수: {len(cases)}  (train={int((cases['split']=='train').sum())}, "
          f"val={int((cases['split']=='val').sum())}, test={int((cases['split']=='test').sum())}, "
          f"external={int((cases['split']=='external').sum())} [tss={external_tss}])")
    print(f"RNA 유전자 수: {rna_input_dim} (top{args.n_genes}, {args.gene_selection} 기준, seed={args.seed})")
    print(f"age_mean={age_mean:.2f} age_std={age_std:.2f} (전체 코호트 기준, train.py 관례와 동일)")

    model = ClinicalRNAOnly(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        use_staging=args.clinical_staging, stage_stats=stage_stats,
    ).to(device)
    model_prefix = f"BRCA_M7_TOP{args.n_genes}"
    if args.gene_selection == "cox":
        model_prefix += "_COXGENE"
    if args.clinical_staging:
        model_prefix += "_STG"
    # M4(scripts/train_brca_m4.py)와 동일 관례 — model_prefix 자체는 fold와 무관하게 유지하고
    # fold_suffix를 파일명 끝에 붙인다(ext_tag가 model_prefix 뒤/_seed 앞에 끼므로 train_light.py
    # 식 "_FOLD{f}OF{n}을 model_prefix에 바로 붙이는" 관례는 못 씀).
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
          f"epochs={cfg.light.epochs} | patience={args.patience} | cox_batch_size={cfg.light.cox_batch_size}")

    run_ts = datetime.now().strftime("%m%d::%H%M")
    group_ts = args.group_ts or run_ts
    wandb_group = f"{model_prefix}_{group_ts}"
    if WANDB_AVAILABLE:
        wandb.init(
            project="Path-ViT",
            name=f"BRCA_{model_prefix}_seed{cfg.light.seed}{fold_suffix}_{run_ts}",
            group=wandb_group,
            config={
                "epochs": cfg.light.epochs, "lr": cfg.light.lr, "weight_decay": cfg.light.weight_decay,
                "seed": cfg.light.seed, "patience": args.patience, "n_genes": args.n_genes,
                "fold": args.fold, "n_folds": args.n_folds,
                "rna_input_dim": rna_input_dim, "model": model_prefix, "dataset": "brca",
            },
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.light.lr, weight_decay=cfg.light.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg)

    ckpt_dir = Path(__file__).parent.parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"survival_brca_best_{model_prefix.lower()}{ext_tag.lower()}_seed{args.seed}{fold_suffix}.pt"

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
        wandb.run.summary["test_hr"] = test_metrics["hr"]
        wandb.run.summary["test_log_rank_p"] = test_metrics["log_rank_p"]
        wandb.run.summary["test_auc_mean"] = test_td_auc["auc_mean"]

    if external_loader is not None:
        # train_brca_m4.py와 동일 관례 — PAAD/CPTAC 구도(train.py --eval-external-ckpt)와 같은
        # CSV 포맷으로 저장해 pool_multiseed_*_preds.py/paired_bootstrap_delta.py 재사용 가능.
        external_metrics = evaluate(model, external_loader, device)
        external_td_auc = compute_time_dependent_auc(
            train_metrics_final["times"], train_metrics_final["events"],
            external_metrics["times"], external_metrics["events"], external_metrics["risks"],
        )
        print(f"\n=== BRCA External Test (institution={external_tss}, best checkpoint) ===")
        print(_log_line("external", external_metrics, external_td_auc))
        import csv
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
            wandb.run.summary["external_hr"] = external_metrics["hr"]
            wandb.run.summary["external_log_rank_p"] = external_metrics["log_rank_p"]
            wandb.run.summary["external_auc_mean"] = external_td_auc["auc_mean"]

    if WANDB_AVAILABLE:
        wandb.finish()

    elapsed = datetime.now() - start_time
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    print(f"\n소요 시간: {h}h {m}m {s}s")


if __name__ == "__main__":
    main()
