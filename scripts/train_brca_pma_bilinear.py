"""
TCGA-BRCA PMA_Bilinear(ViT_PMA_Bilinear) internal 학습/평가 — scripts/train_brca_m4.py의
Bilinear-fusion 버전. train_brca_m4.py와 동일하게 train.py 실제 코드를 그대로 import한다.

[배경] 2026-08-31 — BRCA에서 PORPOISE(RNA-무관 plain ABMIL + Kronecker fusion, seed84)가
0.6759로 나와 기존 PMA(co-attention 기반, seed42) 0.7155보다 오히려 낮았다. PAAD에서는
PORPOISE가 co-attention 계열을 확실히 이겼던 것과 정반대 방향이라, "PMA의 co-attention
pooling 자체(표본이 크면 유리)"와 "PORPOISE의 Kronecker fusion(표본 규모 무관하게 유리할
수도)"이라는 두 요인을 분리해서 봐야 한다는 결론 — 이 스크립트는 **PMA의 WSI pooling(4관점
+ co-attention)은 그대로 두고, 마지막 결합 단계만 concat에서 Kronecker product로 바꾼
하이브리드**(models/vit_pma_bilinear.py::ViT_PMA_Bilinear)를 검증한다.

train_brca_m4.py(0.7155, seed42)와 최대한 같은 조건으로 맞춤:
    backbone=uni, patch-keep-frac=0.8, rna-aux-weight=1.0(기본 원본과 동일), use_coattn=True
달라지는 것(Bilinear fusion의 구조적 제약 때문에 불가피):
    combine_mode: 원본은 "concat"(ClinicalEncoder MLP로 z_clinical 임베딩) → 여기선 항상
    "cox_add"로 고정(clinical은 raw feature 직결 Cox 가산항, PORPOISE와 동일 관례) — Kronecker
    product가 두 벡터 사이 연산이라 세 벡터를 자연스럽게 한 번에 넣을 방법이 없어서, PORPOISE가
    이미 검증한 "clinical은 밖으로 뺀다" 선택을 그대로 따른다. clinical 신호가 age/sex뿐이라
    (BRCA는 staging/margin 미사용) 이 변경 자체의 영향은 이 프로젝트 전체에서 반복 확인된
    "약한 clinical 신호는 raw 직결이 MLP보다 낫거나 같다"는 결론상 미미할 것으로 예상.

사용법:
    python -m scripts.train_brca_pma_bilinear --seed 42
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
from models.vit_pma_bilinear import ViT_PMA_Bilinear
from models.rna_predictor import RNAPredictionHead
from models.clinical_encoder import age_stats_from_csv
from train import (
    set_seed, _build_scheduler, _log_line, train_one_epoch, evaluate, WANDB_AVAILABLE,
)
from utils.metrics import compute_time_dependent_auc
from scripts.brca_common import (
    CLINICAL_PATH, BRCASlideDataset, _identity_collate, load_case_table, load_rna_matrix,
    MANIFEST_PATH, EXTERNAL_TSS,
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-genes", type=int, default=1500)
    parser.add_argument("--patch-keep-frac", type=float, default=0.8)
    parser.add_argument("--rna-aux-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=None, help="cfg.train.epochs(기본 30) 덮어쓰기.")
    parser.add_argument("--group-ts", type=str, default=None)
    parser.add_argument(
        "--external-tss", type=str, default=EXTERNAL_TSS,
        help="train_brca_m4.py --external-tss와 동일. 'none'이면 external 없이 전부 internal 6:2:2.",
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

    gene_path = OUT_DIR / f"selected_genes_top_{args.n_genes}.csv"
    if not gene_path.exists():
        raise FileNotFoundError(
            f"{gene_path} 없음 — 먼저 실행: python -m scripts.select_brca_rna_genes "
            f"--seed {args.seed} --n-genes {args.n_genes}"
        )
    gene_ids = pd.read_csv(gene_path)["gene_id"].tolist()
    rna_input_dim = len(gene_ids)

    cases = load_case_table(args.seed, external_tss=external_tss)
    rna_df = load_rna_matrix(gene_ids)
    manifest = pd.read_csv(MANIFEST_PATH)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATH)
    print(f"case 수: {len(cases)}  (train={int((cases['split']=='train').sum())}, "
          f"val={int((cases['split']=='val').sum())}, test={int((cases['split']=='test').sum())}, "
          f"external={int((cases['split']=='external').sum())} [tss={external_tss}])")
    print(f"RNA 유전자 수: {rna_input_dim} (top{args.n_genes}, 고분산 기준, seed={args.seed})")

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

    model = ViT_PMA_Bilinear(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=True, backbone="uni",
    ).to(device)
    if args.rna_aux_weight > 0:
        model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, rna_input_dim).to(device)

    model_prefix = f"BRCA_PMA_BILINEAR_TOP{args.n_genes}"
    if args.patch_keep_frac < 1.0:
        model_prefix += "_SS"
    if args.rna_aux_weight > 0:
        model_prefix += "_AUX"
    print(f"Model: ViT_PMA_Bilinear (uni backbone, co-attention pooling + Kronecker fusion) | "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"lr={cfg.train.lr:.1e} | weight_decay={cfg.train.weight_decay:.1e} | epochs={cfg.train.epochs} | "
          f"patch_keep_frac={args.patch_keep_frac} | rna_aux_weight={args.rna_aux_weight} | "
          f"cox_batch_size={cfg.train.cox_batch_size}")

    run_ts = datetime.now().strftime("%m%d::%H%M")
    group_ts = args.group_ts or run_ts
    wandb_group = f"{model_prefix}_{group_ts}"
    if WANDB_AVAILABLE:
        wandb.init(
            project="Path-ViT",
            name=f"BRCA_{model_prefix}_seed{cfg.train.seed}_{run_ts}",
            group=wandb_group,
            config={
                "epochs": cfg.train.epochs, "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay,
                "seed": cfg.train.seed, "n_genes": args.n_genes, "rna_input_dim": rna_input_dim,
                "patch_keep_frac": args.patch_keep_frac, "rna_aux_weight": args.rna_aux_weight,
                "embed_dim": cfg.model.embed_dim, "model": model_prefix, "dataset": "brca",
            },
        )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg)

    ckpt_dir = Path(__file__).parent.parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"survival_brca_best_{model_prefix.lower()}{ext_tag.lower()}_seed{args.seed}.pt"

    best_score, best_metrics = -1.0, {}
    for epoch in range(cfg.train.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(
            model, train_loader, optimizer, cfg, device, amp_ctx, None,
            patch_keep_frac=args.patch_keep_frac, rna_aux_weight=args.rna_aux_weight,
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
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_c_index": best_score}, ckpt_path)
            print(f"  -> checkpoint saved (c_index={best_score:.4f}, HR={metrics['hr']:.3f}, "
                  f"log-rank p={metrics['log_rank_p']:.4f})")
            if WANDB_AVAILABLE:
                wandb.run.summary["best_val_c_index"] = best_score
                wandb.run.summary["best_epoch"] = epoch + 1

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
    pred_path = pred_dir / f"brca_{model_prefix}{ext_tag}_seed{args.seed}.csv"
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
        pred_path = pred_dir / f"brca_{model_prefix}{ext_tag}_seed{args.seed}.csv"
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
