"""
사용자 질문(2026-09-05) 후속: ABMIL(model.attn_pool.attn) 게이트가 균등분포로 붕괴한 게
(1) 애초에 초기화 근처에서 한 번도 못 벗어난 것인지, (2) 학습 초반엔 약간 벌어졌다가
weight_decay(기본 1e-1, 꽤 강함)에 밀려 다시 균등으로 눌린 것인지 구분한다. 그리고
(3) attn_pool.attn 파라미터만 weight_decay를 0으로 빼면(--exempt-attn-wd) 실제로 attention이
더 뾰족해지는지도 같은 스크립트로 확인한다(3개 질문을 한 번에: baseline 궤적 + wd 가설 검증).

PAAD(tcga) PMA 1-layer, pdac_consistency_1500+CNV+STG+margin+CLR100+cox_add 레시피(fold0/seed84,
이 세션에서 이미 확인된 recipe)로 직접 학습 루프를 돌리면서(train.py의 train_one_epoch/evaluate를
그대로 재사용) 매 epoch 끝에 고정된 일부 환자(전체 val+test, 셔플 없음)에 대해 attn_weights를
뽑아 정규화 entropy(H(p)/log(N), 1에 가까울수록 완전 균등)를 측정한다.

사용법:
    python -m scripts.diagnose_abmil_attn_training                    # baseline(기존 동작)
    python -m scripts.diagnose_abmil_attn_training --exempt-attn-wd   # attn_pool.attn만 wd=0
"""
import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, pdac_consistency_gene_ids
from models import ViT_PMA
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df
from train import (
    set_seed, train_one_epoch, evaluate, _identity_collate, _make_amp_ctx, _log_line,
)


def _mean_normalized_entropy(model, patients, device, amp_ctx, chunk_size) -> float:
    """patients(list of patient_slides 리스트)에 대해 attn_pool의 attn_weights 정규화
    entropy(H(p)/log(N))를 재서 평균낸다 — 1에 가까울수록 완전 균등(붕괴), 0에 가까울수록
    소수 패치에 집중."""
    captured = {}

    def _hook(module, inputs, output):
        _, attn_weights, _ = output
        captured["w"] = attn_weights.detach()

    handle = model.attn_pool.register_forward_hook(_hook)
    entropies = []
    with torch.no_grad():
        for patient_slides in patients:
            all_feats = torch.cat([s["features"] for s in patient_slides], dim=0).to(device).float()
            with amp_ctx:
                patch_tokens = model.cnn.forward_pooled(all_feats) if hasattr(model.cnn, "forward_pooled") else model.cnn(all_feats)
                coords = None
                if "coords" in patient_slides[0]:
                    coords = torch.cat([s["coords"] for s in patient_slides], dim=0).to(device)
                ctx_tokens = model.vit(patch_tokens, coords) if model.vit is not None else patch_tokens
                _ = model.attn_pool(ctx_tokens)
            w = captured["w"].float()
            n = w.shape[0]
            if n < 2:
                continue
            p = w.clamp(min=1e-12)
            h = -(p * p.log()).sum().item()
            entropies.append(h / np.log(n))
    handle.remove()
    return float(np.mean(entropies)) if entropies else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--backbone", type=str, default="uni2native")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--exempt-attn-wd", action="store_true",
                         help="model.attn_pool.attn 파라미터를 weight_decay=0인 별도 param group으로 뺀다.")
    parser.add_argument("--probe-every", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx()
    cfg = Config()
    set_seed(args.seed)

    rna_gene_ids = pdac_consistency_gene_ids(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS["tcga"])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS["tcga"]))

    ds_kwargs = dict(
        with_clinical=True, with_staging=True, with_margin=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone=args.backbone,
    )
    fold_kwargs = dict(fold=args.fold, n_folds=args.n_folds)
    train_ds = WSISurvivalDataset(cfg.data, dataset="tcga", split="train", **fold_kwargs, **ds_kwargs)
    val_ds   = WSISurvivalDataset(cfg.data, dataset="tcga", split="val",   **fold_kwargs, **ds_kwargs)
    test_ds  = WSISurvivalDataset(cfg.data, dataset="tcga", split="test",  **fold_kwargs, **ds_kwargs)
    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    probe_patients = [p for p in DataLoader(val_ds, shuffle=False, **dl_kwargs) if p] + \
                     [p for p in DataLoader(test_ds, shuffle=False, **dl_kwargs) if p]
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} probe={len(probe_patients)}")

    rna_input_dim = len(rna_gene_ids) + 8
    model = ViT_PMA(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=cfg.data.precomputed, backbone=args.backbone, combine_mode="cox_add",
        use_staging=True, stage_stats=stage_stats, use_margin=True, margin_stats=margin_stats,
    ).to(device)

    attn_params = list(model.attn_pool.attn.parameters())
    attn_param_ids = {id(p) for p in attn_params}
    from train import _branch_param_groups
    groups = _branch_param_groups(model)
    non_attn_wd_params = [p for p in groups["wsi"] + groups["other"] if id(p) not in attn_param_ids]
    if args.exempt_attn_wd:
        print("attn_pool.attn 파라미터를 weight_decay=0으로 분리합니다.")
        optimizer = torch.optim.AdamW([
            {"params": groups["clinical"], "lr": cfg.train.lr * 100.0, "weight_decay": cfg.train.weight_decay},
            {"params": groups["rna"], "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay},
            {"params": non_attn_wd_params, "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay},
            {"params": attn_params, "lr": cfg.train.lr, "weight_decay": 0.0},
        ])
    else:
        optimizer = torch.optim.AdamW([
            {"params": groups["clinical"], "lr": cfg.train.lr * 100.0, "weight_decay": cfg.train.weight_decay},
            {"params": groups["rna"], "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay},
            {"params": groups["wsi"] + groups["other"], "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay},
        ])

    cfg.train.epochs = args.epochs
    from train import _build_scheduler
    scheduler = _build_scheduler(optimizer, cfg)

    init_entropy = _mean_normalized_entropy(model, probe_patients, device, amp_ctx, cfg.train.cnn_chunk_size)
    print(f"epoch 0(초기화 직후) 정규화 entropy(1=완전균등) = {init_entropy:.4f}")

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, cfg, device, amp_ctx, None, rna_aux_weight=0.0)
        scheduler.step()
        if (epoch + 1) % args.probe_every == 0 or epoch == args.epochs - 1:
            ent = _mean_normalized_entropy(model, probe_patients, device, amp_ctx, cfg.train.cnn_chunk_size)
            metrics = evaluate(model, val_loader, cfg, device, amp_ctx, None)
            print(f"epoch {epoch+1:3d} | loss={loss:.4f} | val_c_index={metrics['c_index']:.4f} | "
                  f"attn 정규화 entropy={ent:.4f}")

    print(f"\n=== exempt_attn_wd={args.exempt_attn_wd} 최종 정리 ===")
    print(f"초기 entropy={init_entropy:.4f} -> 최종 entropy={ent:.4f}")


if __name__ == "__main__":
    main()
