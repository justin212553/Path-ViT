"""
TCGA-BRCA PORPOISE(ViT_PORPOISE, no_aux 레시피) internal 학습/평가 — scripts/train_brca_m4.py의
PORPOISE 버전. train_brca_m4.py와 마찬가지로 train.py 실제 코드(train_one_epoch/evaluate/
_build_scheduler)를 그대로 import해서 쓴다 — 새로 재구현하면 로직이 미묘하게 어긋날 위험이 있다.

[배경] 2026-08-31 PAAD(TCGA-PAAD, N=152)에서 진행한 architecture search 전체(MCAT의
multi-pathway co-attention → PORPOISE의 Kronecker/bilinear fusion) 도중, 사용자가 "이거
전부 BRCA에서 돌아가는 줄 알았다"는 걸 뒤늦게 발견 — 실제로는 전부 PAAD였고, PORPOISE가
PAAD에서 M7(WSI 없음) 대비 paired bootstrap으로 완전히 비유의(seed84+126, internal delta
0.0000/p=0.977, external delta -0.0042/p=0.803)로 나왔다. "코호트가 크면 WSI 노이즈 문제가
풀릴 것"이라는 원래 가설(2026-07-22 최상위 발견 — PMA가 BRCA 스케일에서 M7 대비 +0.0535)이
PORPOISE에는 아직 한 번도 테스트된 적이 없다는 뜻이라, 이 스크립트로 그 빈 칸을 채운다.

[2026-09-06 확장] PAAD에서 --cluster-pool 대신 PORPOISE(ABMIL+Kronecker) 아키텍처가 최종
레시피로 자리잡음에 따라, BRCA도 train_brca_m4.py의 --cluster-pool+CLR100 조합(M7을 처음
이긴 기록, test_c_index=0.7539 vs M7 0.7367)과 apple-to-apple 비교가 되도록 gene-selection/
clinical-staging/clinical-lr-mult/k-fold(--fold/--n-folds)를 train_brca_m4.py와 동일하게
포팅했다. CNV/mutation은 BRCA 쪽 데이터/코드 자체가 아직 없어(PDAC 전용, 2026-09-05 확인)
포함하지 않는다 — --surv-loss(nll_surv/both)도 이 스크립트엔 아직 이식 안 함(사용자가 요청한
범위는 "cluster_pool 대신 PORPOISE 아키텍처"까지).

PAAD의 no_aux 최종 레시피와 동일하게 유지:
    ViT_PORPOISE(models/vit_porpoise.py), BilinearFusion(Kronecker product), 기본 gated-ABMIL
    (RNA 무관 pooling), combine_mode는 내부적으로 항상 cox_add
    --attn-dispersion 켬(PAAD ablation에서 PORPOISE 성능에 크게 기여한 것으로 확인됨)
    --patch-keep-frac 0.8, backbone=uni(embed_dim=64, num_heads=2, num_landmarks=128 — BRCA
    쪽엔 uni2native 리타일링이 아직 없어 train_brca_m4.py와 동일하게 uni 그대로 유지)
BRCA라서 train_brca_m4.py와 동일하게 맞춘 것:
    --gene-selection {variance,cox,literature,literature_categorized,consistency}
    --clinical-staging, --clinical-lr-mult, --fold/--n-folds(k-fold, brca_common.py)
    rna_aux_head: 안 씀(PAAD ablation에서 소폭 해로운 것으로 확인, PORPOISE no_aux 레시피 그대로)

사용법:
    python -m scripts.train_brca_porpoise --seed 84 --gene-selection consistency --clinical-staging \\
        --clinical-lr-mult 100 --external-tss none --fold 0 --n-folds 5
"""
import argparse
import math
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
from models.vit_porpoise import ViT_PORPOISE
from models.clinical_encoder import age_stats_from_csv, stage_stats_from_csv
from train import (
    set_seed, _build_scheduler, _log_line, train_one_epoch, evaluate, WANDB_AVAILABLE,
    _branch_param_groups,
)
from utils.metrics import compute_time_dependent_auc
from scripts.brca_common import (
    CLINICAL_PATH, BRCASlideDataset, _identity_collate, load_case_table, load_case_table_kfold,
    load_rna_matrix, load_rna_matrix_categorized, load_literature_categories, MANIFEST_PATH, EXTERNAL_TSS,
)

if WANDB_AVAILABLE:
    import wandb

OUT_DIR = Path("data/brca_rna_gene_selection")


def _make_amp_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--n-genes", type=int, default=1500)
    parser.add_argument("--gene-selection", type=str, default="variance",
                         choices=["variance", "cox", "literature", "literature_categorized", "consistency"],
                         help="train_brca_m4.py --gene-selection과 동일 관례.")
    parser.add_argument("--fdr-threshold", type=float, default=None,
                         help="--gene-selection cox 전용 — BH-FDR q값 컷오프.")
    parser.add_argument("--clinical-staging", action="store_true",
                         help="train_brca_m4.py --clinical-staging 이식 — ClinicalEncoder 대신 "
                              "ViT_PORPOISE의 cox_add raw-feature staging 입력(models/vit_porpoise.py).")
    parser.add_argument("--clinical-lr-mult", type=float, default=1.0,
                         help="train_brca_m4.py --clinical-lr-mult 이식 — clinical_linear(cox_add "
                              "가산항) 브랜치 lr 배율.")
    parser.add_argument("--patch-keep-frac", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=None, help="cfg.train.epochs(기본 30) 덮어쓰기.")
    parser.add_argument("--early-stop-patience", type=int, default=None,
                         help="train_brca_m4.py --early-stop-patience 이식.")
    parser.add_argument("--group-ts", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None,
                         help="train_brca_m4.py --fold 이식 — 주어지면(0-based) 기존 단일 6:2:2 "
                              "대신 K-fold(scripts/brca_common.py::load_case_table_kfold)를 쓴다.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--external-tss", type=str, default=EXTERNAL_TSS,
        help="train_brca_m4.py --external-tss와 동일 — institution-level external holdout. "
             "'none'이면 external 없이 전부 internal 6:2:2.",
    )
    args = parser.parse_args()
    external_tss = None if args.external_tss.lower() == "none" else args.external_tss
    ext_tag = f"_EXTTSS{external_tss}" if external_tss else ""

    cfg = Config()
    cfg.data.seed = cfg.train.seed = args.seed
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx(device)
    start_time = datetime.now()

    # --- RNA 유전자셋 선택(train_brca_m4.py와 동일 관례) ---
    gene_ids = None
    if args.gene_selection == "literature_categorized":
        if args.fdr_threshold is not None:
            raise ValueError("--fdr-threshold는 --gene-selection cox와 함께만 쓸 수 있습니다.")
    elif args.gene_selection == "literature":
        if args.fdr_threshold is not None:
            raise ValueError("--fdr-threshold는 --gene-selection cox와 함께만 쓸 수 있습니다.")
        gene_path = Path("data/brca_rna_gene_selection_literature/selected_genes.csv")
        select_hint = "python -m scripts.select_brca_rna_genes_literature"
    elif args.gene_selection == "consistency":
        if args.fdr_threshold is not None:
            raise ValueError("--fdr-threshold는 --gene-selection cox와 함께만 쓸 수 있습니다.")
        gene_path = Path("data/brca_rna_gene_selection_consistency/selected_genes.csv")
        select_hint = "python -m scripts.select_brca_rna_genes_consistency"
    else:
        gene_dir = OUT_DIR if args.gene_selection == "variance" else Path("data/brca_rna_gene_selection_cox")
        if args.fdr_threshold is not None:
            if args.gene_selection != "cox":
                raise ValueError("--fdr-threshold는 --gene-selection cox와 함께만 쓸 수 있습니다.")
            gene_path = gene_dir / f"selected_genes_fdr{args.fdr_threshold:g}.csv"
            select_hint = f"python -m scripts.select_brca_rna_genes_cox --seed {args.seed} --fdr-threshold {args.fdr_threshold:g}"
        else:
            gene_path = gene_dir / f"selected_genes_top_{args.n_genes}.csv"
            select_module = "select_brca_rna_genes" if args.gene_selection == "variance" else "select_brca_rna_genes_cox"
            select_hint = f"python -m scripts.{select_module} --seed {args.seed} --n-genes {args.n_genes}"

    if args.gene_selection == "literature_categorized":
        literature_categories = load_literature_categories()
        rna_input_dim = len(literature_categories)
    else:
        if not gene_path.exists():
            raise FileNotFoundError(f"{gene_path} 없음 — 먼저 실행: {select_hint}")
        gene_ids = pd.read_csv(gene_path)["gene_id"].tolist()
        rna_input_dim = len(gene_ids)
    stage_stats = stage_stats_from_csv(CLINICAL_PATH) if args.clinical_staging else None

    if args.fold is not None:
        cases = load_case_table_kfold(args.seed, args.fold, args.n_folds, external_tss=external_tss)
    else:
        cases = load_case_table(args.seed, external_tss=external_tss)
    if args.gene_selection == "literature_categorized":
        rna_df = load_rna_matrix_categorized(literature_categories)
    else:
        rna_df = load_rna_matrix(gene_ids)
    manifest = pd.read_csv(MANIFEST_PATH)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATH)
    print(f"case 수: {len(cases)}  (train={int((cases['split']=='train').sum())}, "
          f"val={int((cases['split']=='val').sum())}, test={int((cases['split']=='test').sum())}, "
          f"external={int((cases['split']=='external').sum())} [tss={external_tss}])")
    if args.gene_selection == "literature_categorized":
        gene_tag = f"LITCAT{rna_input_dim}"
    elif args.gene_selection == "literature":
        gene_tag = f"LIT{rna_input_dim}"
    elif args.gene_selection == "consistency":
        gene_tag = f"CONS{rna_input_dim}"
    elif args.fdr_threshold is not None:
        gene_tag = f"FDR{args.fdr_threshold:g}"
    else:
        gene_tag = f"TOP{args.n_genes}"
    print(f"RNA 유전자 수: {rna_input_dim} ({gene_tag}, {args.gene_selection} 기준, seed={args.seed})")

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    stg = args.clinical_staging
    train_ds     = BRCASlideDataset(cases[cases["split"] == "train"],    rna_df, manifest, with_staging=stg)
    val_ds       = BRCASlideDataset(cases[cases["split"] == "val"],      rna_df, manifest, with_staging=stg)
    test_ds      = BRCASlideDataset(cases[cases["split"] == "test"],     rna_df, manifest, with_staging=stg)
    external_ds  = BRCASlideDataset(cases[cases["split"] == "external"], rna_df, manifest, with_staging=stg) if external_tss else None
    train_loader      = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **dl_kwargs)
    val_loader        = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader       = DataLoader(test_ds,  shuffle=False, **dl_kwargs)
    external_loader   = DataLoader(external_ds, shuffle=False, **dl_kwargs) if external_ds is not None else None

    # PAAD no_aux 최종 레시피와 동일 — 기본 gated-ABMIL(use_meanpool/use_coattn 둘 다 False),
    # attn-dispersion 켬. rna_aux_head는 안 붙인다(PAAD ablation에서 소폭 해로운 것으로 확인).
    model = ViT_PORPOISE(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=True, backbone="uni", use_attn_dispersion=True,
        use_staging=args.clinical_staging, stage_stats=stage_stats,
    ).to(device)

    model_prefix = f"BRCA_PORPOISE_{gene_tag}_SS_DISP"
    if args.gene_selection == "cox":
        model_prefix += "_COXGENE"
    if args.clinical_staging:
        model_prefix += "_STG"
    if args.early_stop_patience is not None:
        model_prefix += f"_ES{args.early_stop_patience}"
    if args.clinical_lr_mult != 1.0:
        model_prefix += f"_CLR{args.clinical_lr_mult:g}"
    fold_suffix = f"_fold{args.fold}of{args.n_folds}" if args.fold is not None else ""
    print(f"Model: ViT_PORPOISE (uni backbone, gated-ABMIL+BilinearFusion, "
          f"use_attn_dispersion=True) | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"lr={cfg.train.lr:.1e} | weight_decay={cfg.train.weight_decay:.1e} | epochs={cfg.train.epochs} | "
          f"patch_keep_frac={args.patch_keep_frac} | cox_batch_size={cfg.train.cox_batch_size}")

    run_ts = datetime.now().strftime("%m%d::%H%M")
    group_ts = args.group_ts or run_ts
    wandb_group = f"{model_prefix}_{group_ts}"
    if WANDB_AVAILABLE:
        wandb.init(
            project="Path-ViT",
            name=f"BRCA_{model_prefix}_seed{cfg.train.seed}{fold_suffix}_{run_ts}",
            group=wandb_group,
            config={
                "epochs": cfg.train.epochs, "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay,
                "seed": cfg.train.seed, "n_genes": args.n_genes, "rna_input_dim": rna_input_dim,
                "patch_keep_frac": args.patch_keep_frac, "embed_dim": cfg.model.embed_dim,
                "clinical_lr_mult": args.clinical_lr_mult,
                "fold": args.fold, "n_folds": args.n_folds,
                "model": model_prefix, "dataset": "brca",
            },
        )

    if args.clinical_lr_mult != 1.0:
        groups = _branch_param_groups(model)
        param_groups = []
        if groups["clinical"]:
            param_groups.append({"params": groups["clinical"], "lr": cfg.train.lr * args.clinical_lr_mult})
        if groups["rna"]:
            param_groups.append({"params": groups["rna"], "lr": cfg.train.lr})
        if groups["other"]:
            param_groups.append({"params": groups["other"], "lr": cfg.train.lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
        print(f"branch-lr-mult 적용: clinical={args.clinical_lr_mult}x({len(groups['clinical'])}개 텐서), "
              f"other=1x({len(groups['other']) + len(groups['rna'])}개 텐서)")
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        )
    scheduler = _build_scheduler(optimizer, cfg)

    ckpt_dir = Path(__file__).parent.parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"survival_brca_best_{model_prefix.lower()}{ext_tag.lower()}_seed{args.seed}{fold_suffix}.pt"

    best_score, best_metrics = -1.0, {}
    epochs_since_improve = 0
    for epoch in range(cfg.train.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(
            model, train_loader, optimizer, cfg, device, amp_ctx, None,
            patch_keep_frac=args.patch_keep_frac, rna_aux_weight=0.0,
        )
        train_metrics = evaluate(model, train_eval_loader, cfg, device, amp_ctx, None)
        metrics = evaluate(model, val_loader, cfg, device, amp_ctx, None)
        val_td_auc = compute_time_dependent_auc(
            train_metrics["times"], train_metrics["events"], metrics["times"], metrics["events"], metrics["risks"],
        )
        scheduler.step()

        c_index = metrics.get("c_index", float("nan"))
        score = c_index if not math.isnan(c_index) else -1.0
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
            epochs_since_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_c_index": best_score}, ckpt_path)
            print(f"  -> checkpoint saved (c_index={best_score:.4f}, HR={metrics['hr']:.3f}, "
                  f"log-rank p={metrics['log_rank_p']:.4f})")
            if WANDB_AVAILABLE:
                wandb.run.summary["best_val_c_index"] = best_score
                wandb.run.summary["best_epoch"] = epoch + 1
        else:
            epochs_since_improve += 1
            if (args.early_stop_patience is not None
                    and epochs_since_improve >= args.early_stop_patience):
                print(f"  -> early stop: 최근 {epochs_since_improve} epoch 동안 val c_index 갱신 없음 "
                      f"(best epoch {best_metrics.get('epoch', '-')}, best c_index={best_score:.4f})")
                break

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
    if WANDB_AVAILABLE:
        wandb.run.summary["test_c_index"] = test_metrics["c_index"]
        wandb.run.summary["test_hr"] = test_metrics["hr"]
        wandb.run.summary["test_log_rank_p"] = test_metrics["log_rank_p"]
        wandb.run.summary["test_auc_mean"] = test_td_auc["auc_mean"]

    if external_loader is not None:
        external_metrics = evaluate(model, external_loader, cfg, device, amp_ctx, None)
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
