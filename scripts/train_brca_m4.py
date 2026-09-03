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
from models.clinical_encoder import age_stats_from_csv, stage_stats_from_csv
from train import (
    set_seed, _build_scheduler, _log_line, train_one_epoch, evaluate, WANDB_AVAILABLE,
    _branch_param_groups,
)
from utils.metrics import compute_time_dependent_auc
from scripts.brca_common import (
    CLINICAL_PATH, BRCASlideDataset, _identity_collate, load_case_table, load_case_table_kfold,
    load_rna_matrix, MANIFEST_PATH, EXTERNAL_TSS,
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
    parser.add_argument("--gene-selection", type=str, default="variance", choices=["variance", "cox"],
                         help="2026-09-02: 'variance'(기본, data/brca_rna_gene_selection/) 또는 "
                              "'cox'(생존 라벨 기반 univariate Cox score test, "
                              "data/brca_rna_gene_selection_cox/, "
                              "scripts/select_brca_rna_genes_cox.py) 중 선택 — train_brca_m7.py와 "
                              "동일 관례, M4/M7 비교 시 반드시 동일 값을 써야 함.")
    parser.add_argument("--fdr-threshold", type=float, default=None,
                         help="train_brca_m7.py --fdr-threshold와 동일 — --gene-selection cox와 "
                              "함께 주어지면 top-N 대신 BH-FDR q<threshold 패널을 쓴다.")
    parser.add_argument("--clinical-staging", action="store_true",
                         help="2026-09-02: ClinicalEncoder(ViT_PMA 내장) 입력에 AJCC 병기를 "
                              "추가한다(train.py --clinical-staging과 동일 관례). "
                              "train_brca_m7.py --clinical-staging과 동일 관례 — M4/M7 비교 시 "
                              "반드시 동일 값을 써야 함.")
    parser.add_argument("--patch-keep-frac", type=float, default=0.8)
    parser.add_argument("--rna-aux-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=None, help="cfg.train.epochs(기본 30) 덮어쓰기.")
    parser.add_argument("--no-spatial-embed", action="store_true",
                         help="train.py --no-spatial-embed와 동일 — SpatialPositionEmbedding(좌표 "
                              "sin/cos 인코딩)을 끈다. PAAD에서는 null이었지만 WSI 신호 자체가 "
                              "없던 환경이라(findings_backlog.md), WSI가 유의미해진 BRCA에서 재검증.")
    parser.add_argument("--rel-bias-attention", action="store_true",
                         help="train.py --rel-bias-attention과 동일 — 절대좌표 SpatialPositionEmbedding "
                              "대신 상대offset(Δrow,Δcol) attention bias(Swin류, models/vit_encoder.py"
                              "::RelativeBiasFullAttention)를 넣은 전체(O(N^2)) attention으로 교체한다 "
                              "(use_nystrom/use_spatial_embed 자동 False). 2026-07-23: 얼린 Stage1+잔차 "
                              "branch(models/spatial_residual.py)가 PAAD·BRCA 둘 다에서 실패한 뒤 "
                              "WSI branch 안에서 처음부터 end-to-end로 학습시키는 대안 검증용.")
    parser.add_argument("--knn-bias-attention", action="store_true",
                         help="train.py --knn-bias-attention과 동일 — --rel-bias-attention의 희소 "
                              "버전(models/vit_encoder.py::KNNBiasAttention, kNN 이웃 k개에만 "
                              "attention). 2026-07-23: BRCA는 슬라이드당 패치 수 중앙값 10,309/최대 "
                              "67,268이라 dense(--rel-bias-attention)가 즉시 CUDA OOM나서 대신 이걸 "
                              "쓴다(use_nystrom/use_spatial_embed 자동 False).")
    parser.add_argument("--knn-k", type=int, default=8,
                         help="--knn-bias-attention 사용 시 패치당 kNN 이웃 수(기본 8).")
    parser.add_argument("--clinical-lr-mult", type=float, default=1.0,
                         help="train.py --clinical-lr-mult 이식 — clinical_encoder 파라미터그룹 lr을 "
                              "base lr의 이 배수로. PAAD에서 20x가 branch-competition을 해소한 효과가 "
                              "BRCA(표본 7배)에서도 재현되는지 확인.")
    parser.add_argument("--rna-lr-mult", type=float, default=1.0,
                         help="train.py --rna-lr-mult 이식 — rna_encoder 파라미터그룹 lr 배수.")
    parser.add_argument("--wsi-extra-mlp", action="store_true",
                         help="train.py --wsi-extra-mlp 이식(models/vit_m1.py::ViT_M1, ViT_PMA가 상속) "
                              "— patch_tokens에 잔차 MLP 한 층 추가.")
    parser.add_argument("--group-ts", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None,
                         help="2026-09-01: 주어지면(0-based) 기존 단일 6:2:2 대신 PAAD와 동일한 "
                              "K-fold(data/dataset.py::_kfold_case_split 방법론, "
                              "scripts/brca_common.py::load_case_table_kfold)를 쓴다 — fold 배정 "
                              "자체는 --seed로 셔플되므로(모델 init seed와 동일 값), 진짜 다시드 "
                              "검증(--seed 84/126 x --fold 0..4)을 하려면 fold 배정도 함께 "
                              "바뀐다(PAAD paper-spec 프로토콜과 동일 관례).")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--external-tss", type=str, default=EXTERNAL_TSS,
        help=f"institution-level external holdout(TCGA barcode 2번째 세그먼트, 기본 "
             f"{EXTERNAL_TSS!r} — 공통 case 1058명 중 가장 큰 단일 기관 142명). "
             "'none'이면 external 없이 기존 동작(전부 internal 6:2:2)으로 되돌아간다. "
             "M7과 반드시 동일 값을 써야 비교가 성립한다(scripts/brca_common.py 참조).",
    )
    args = parser.parse_args()
    external_tss = None if args.external_tss.lower() == "none" else args.external_tss
    ext_tag = f"_EXTTSS{external_tss}" if external_tss else ""  # None이면 파일명에 접미사 없음

    cfg = Config()
    cfg.data.seed = cfg.train.seed = args.seed
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.no_spatial_embed:
        cfg.model.use_spatial_embed = False
    if args.rel_bias_attention:
        cfg.model.use_rel_bias_attn = True
        cfg.model.use_nystrom = False
        cfg.model.use_spatial_embed = False
    if args.knn_bias_attention:
        cfg.model.use_knn_bias_attn = True
        cfg.model.knn_attn_k = args.knn_k
        cfg.model.use_nystrom = False
        cfg.model.use_spatial_embed = False
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx(device)
    start_time = datetime.now()

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
    if not gene_path.exists():
        raise FileNotFoundError(f"{gene_path} 없음 — 먼저 실행: {select_hint}")
    gene_ids = pd.read_csv(gene_path)["gene_id"].tolist()
    rna_input_dim = len(gene_ids)
    stage_stats = stage_stats_from_csv(CLINICAL_PATH) if args.clinical_staging else None

    if args.fold is not None:
        cases = load_case_table_kfold(args.seed, args.fold, args.n_folds, external_tss=external_tss)
    else:
        cases = load_case_table(args.seed, external_tss=external_tss)
    rna_df = load_rna_matrix(gene_ids)
    manifest = pd.read_csv(MANIFEST_PATH)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATH)
    print(f"case 수: {len(cases)}  (train={int((cases['split']=='train').sum())}, "
          f"val={int((cases['split']=='val').sum())}, test={int((cases['split']=='test').sum())}, "
          f"external={int((cases['split']=='external').sum())} [tss={external_tss}])")
    gene_tag = f"FDR{args.fdr_threshold:g}" if args.fdr_threshold is not None else f"TOP{args.n_genes}"
    print(f"RNA 유전자 수: {rna_input_dim} ({gene_tag}, {args.gene_selection} 기준, seed={args.seed})")
    print(f"age_mean={age_mean:.2f} age_std={age_std:.2f} (전체 코호트 기준, train.py 관례와 동일)")

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

    model = ViT_PMA(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=True, backbone="uni", use_wsi_extra_mlp=args.wsi_extra_mlp,
        use_staging=args.clinical_staging, stage_stats=stage_stats,
    ).to(device)
    if args.rna_aux_weight > 0:
        model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, rna_input_dim).to(device)

    model_prefix = f"BRCA_PMA_{gene_tag}"
    if args.gene_selection == "cox":
        model_prefix += "_COXGENE"
    if args.clinical_staging:
        model_prefix += "_STG"
    if args.patch_keep_frac < 1.0:
        model_prefix += "_SS"
    if args.rna_aux_weight > 0:
        model_prefix += "_AUX"
    if args.no_spatial_embed:
        model_prefix += "_NOSPATIAL"
    if args.rel_bias_attention:
        model_prefix += "_RELBIAS"
    if args.knn_bias_attention:
        model_prefix += "_KNNATTN"
    if args.wsi_extra_mlp:
        model_prefix += "_XMLP"
    if args.clinical_lr_mult != 1.0:
        model_prefix += f"_CLR{args.clinical_lr_mult:g}"
    if args.rna_lr_mult != 1.0:
        model_prefix += f"_RLR{args.rna_lr_mult:g}"
    # 2026-09-01: PAAD(train_light.py)와 달리 _FOLD{f}OF{n}을 model_prefix 자체엔 안 붙인다
    # (BRCA는 ext_tag가 항상 model_prefix 뒤/_seed 앞에 끼어들어 pool_multiseed_kfold_preds.py의
    # 기본 조회 패턴과 어긋나므로) — 대신 fold_suffix를 파일명 끝(ckpt/pred_path 전부)에
    # 붙여 pool 스크립트의 폴백 패턴({dataset}_{model}_seed{seed}_fold{fold}of{n_folds}.csv)과
    # 정확히 맞춘다. model_prefix 자체는 fold와 무관하게 동일하게 유지.
    fold_suffix = f"_fold{args.fold}of{args.n_folds}" if args.fold is not None else ""
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
            name=f"BRCA_{model_prefix}_seed{cfg.train.seed}{fold_suffix}_{run_ts}",
            group=wandb_group,
            config={
                "epochs": cfg.train.epochs, "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay,
                "seed": cfg.train.seed, "n_genes": args.n_genes, "rna_input_dim": rna_input_dim,
                "patch_keep_frac": args.patch_keep_frac, "rna_aux_weight": args.rna_aux_weight,
                "embed_dim": cfg.model.embed_dim, "use_nystrom": cfg.model.use_nystrom,
                "use_spatial_embed": cfg.model.use_spatial_embed,
                "use_rel_bias_attn": cfg.model.use_rel_bias_attn,
                "use_knn_bias_attn": cfg.model.use_knn_bias_attn,
                "wsi_extra_mlp": args.wsi_extra_mlp,
                "clinical_lr_mult": args.clinical_lr_mult, "rna_lr_mult": args.rna_lr_mult,
                "fold": args.fold, "n_folds": args.n_folds,
                "model": model_prefix, "dataset": "brca",
            },
        )

    if args.clinical_lr_mult != 1.0 or args.rna_lr_mult != 1.0:
        groups = _branch_param_groups(model)
        param_groups = []
        if groups["clinical"]:
            param_groups.append({"params": groups["clinical"], "lr": cfg.train.lr * args.clinical_lr_mult})
        if groups["rna"]:
            param_groups.append({"params": groups["rna"], "lr": cfg.train.lr * args.rna_lr_mult})
        if groups["other"]:
            param_groups.append({"params": groups["other"], "lr": cfg.train.lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
        print(f"branch-lr-mult 적용: clinical={args.clinical_lr_mult}x({len(groups['clinical'])}개 텐서), "
              f"rna={args.rna_lr_mult}x({len(groups['rna'])}개 텐서), other=1x({len(groups['other'])}개 텐서)")
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
        # PAAD/CPTAC 구도(train.py --eval-external-ckpt)와 동일 관례로 CSV까지 저장 —
        # scripts/pool_multiseed_kfold_preds.py류/paired_bootstrap_delta.py를 그대로 재사용할
        # 수 있게. BRCA는 institution 기준 external이라 seed/fold 개념이 없어(단일 실행)
        # 파일명에 fold 표기를 생략한다.
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
