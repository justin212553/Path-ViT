"""
TCGA-BRCA M4(ViT_PMA, PMA_EX_SS_AUX 레시피) internal 학습/평가 — train.py --PMA의 BRCA 버전.

PMA_EX_SS_AUX(우리 프로젝트에서 지금까지 가장 나은 M4 변형 — findings_backlog.md 참조)를
"WSI가 표본을 늘리면 순증분 기여를 하는가"를 검증하기 위해 TCGA-BRCA(1058 case, TCGA-PAAD의
약 7배)로 재현한다. 아키텍처/학습 루프는 train.py 실제 코드를 그대로 import해서 쓴다 — 새로
재구현하면 로직이 미묘하게 어긋날 위험이 있다(reference_repro_m4.py가 레퍼런스 코드를 직접
import하는 것과 같은 원칙).

PMA_EX_SS_AUX와 동일하게 유지하는 것:
    --PMA(ViT_PMA, Nystromformer 공간 컨텍스트 블록 포함 — 사용자 지시: "Nystrom 당연히 쓴다")
    --patch-keep-frac 0.8 (PatchDropout, _SS)
    --rna-aux-weight 1.0 (RNAPredictionHead 보조과제, _AUX)
    backbone=uni, embed_dim=64, num_heads=2, num_transformer_layers=1, num_landmarks=128,
    lr=1e-5, weight_decay=1e-1, epochs=30, warmup_epochs=3, cox_batch_size=16
BRCA라서 바꾼 것:
    --rna-genes: literature_1500(PDAC 전용 subtype 큐레이션) 대신 scripts/select_rnaseq_genes.py
    스타일의 고분산 상위 1500개(scripts/select_brca_rna_genes.py, scripts/extract_brca_rna.py
    docstring 참조) — "_EX" 자리에 해당하지만 PDAC literature curation이 아니므로 접미사는
    "TOP1500"으로 구분한다.
    case 목록/split: scripts/brca_common.py (M7과 반드시 동일 --seed로 비교해야 함)

[좌표 정규화] scripts/brca_common.py::BRCASlideDataset._grid_coords 참조 — HF 다운로드
coords.pt는 픽셀 좌표라 그대로 SpatialPositionEmbedding에 넣을 수 없어 슬라이드 내부
순위로 변환한다.

사용법:
    python -m scripts.train_brca_m4 --seed 42
    python -m scripts.train_brca_m4 --seed 42 --n-genes 1500 --patch-keep-frac 0.8 --rna-aux-weight 1.0
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
from models.vit_pma import ViT_PMA
from models.rna_predictor import RNAPredictionHead
from models.clinical_encoder import age_stats_from_csv
from train import (
    set_seed, _build_scheduler, _log_line, train_one_epoch, evaluate, WANDB_AVAILABLE,
)
from utils.metrics import compute_time_dependent_auc
from scripts.brca_common import (
    CLINICAL_PATH, BRCASlideDataset, _identity_collate, load_case_table, load_rna_matrix,
    MANIFEST_PATH,
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
    parser.add_argument("--no-spatial-embed", action="store_true",
                         help="train.py --no-spatial-embed와 동일 — SpatialPositionEmbedding(좌표 "
                              "sin/cos 인코딩)을 끈다. PAAD에서는 null이었지만 WSI 신호 자체가 "
                              "없던 환경이라(findings_backlog.md), WSI가 유의미해진 BRCA에서 재검증.")
    parser.add_argument("--group-ts", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    cfg.data.seed = cfg.train.seed = args.seed
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.no_spatial_embed:
        cfg.model.use_spatial_embed = False
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

    cases = load_case_table(args.seed)
    rna_df = load_rna_matrix(gene_ids)
    manifest = pd.read_csv(MANIFEST_PATH)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATH)
    print(f"case 수: {len(cases)}  (train={int((cases['split']=='train').sum())}, "
          f"val={int((cases['split']=='val').sum())}, test={int((cases['split']=='test').sum())})")
    print(f"RNA 유전자 수: {rna_input_dim} (top{args.n_genes}, 고분산 기준, seed={args.seed})")
    print(f"age_mean={age_mean:.2f} age_std={age_std:.2f} (전체 코호트 기준, train.py 관례와 동일)")

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    train_ds = BRCASlideDataset(cases[cases["split"] == "train"], rna_df, manifest)
    val_ds   = BRCASlideDataset(cases[cases["split"] == "val"],   rna_df, manifest)
    test_ds  = BRCASlideDataset(cases[cases["split"] == "test"],  rna_df, manifest)
    train_loader      = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **dl_kwargs)
    val_loader        = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader       = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    model = ViT_PMA(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=True, backbone="uni",
    ).to(device)
    if args.rna_aux_weight > 0:
        model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, rna_input_dim).to(device)

    model_prefix = f"BRCA_PMA_TOP{args.n_genes}"
    if args.patch_keep_frac < 1.0:
        model_prefix += "_SS"
    if args.rna_aux_weight > 0:
        model_prefix += "_AUX"
    if args.no_spatial_embed:
        model_prefix += "_NOSPATIAL"
    print(f"Model: ViT_PMA (uni backbone, use_nystrom={cfg.model.use_nystrom}, "
          f"use_spatial_embed={cfg.model.use_spatial_embed}) | "
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
                "embed_dim": cfg.model.embed_dim, "use_nystrom": cfg.model.use_nystrom,
                "use_spatial_embed": cfg.model.use_spatial_embed, "model": model_prefix, "dataset": "brca",
            },
        )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg)

    ckpt_dir = Path(__file__).parent.parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"survival_brca_best_{model_prefix.lower()}_seed{args.seed}.pt"

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
