"""
scripts/diagnose_porpoise_ctxtoken_diversity.py — "attn_pool의 최종 가중치 부여 방식 자체는
쓸모없다(§2-4, diagnose_porpoise_residual_hazard.py)"와 "attn_pool을 통째로 MeanPooling으로
바꿔 재학습하면 성능이 떨어진다(§2-1, 0.7119→0.6784)"가 서로 모순돼 보이는 지점을 직접
검증한다.

[가설] 두 결과는 서로 다른 걸 測정했다 — §2-4는 "같은 ctx_tokens(self.vit 출력, pooling
직전) 위에서 가중합 vs 단순평균"만 비교했다(ctx_tokens 고정). §2-1의 재학습 비교는 attn_pool
이라는 학습 가능한 모듈이 애초에 존재하느냐 자체가 self.vit(나이스트롬 블록)이 만들어내는
ctx_tokens 자체를 다르게 학습시켰을 수 있다 — attn_pool의 최종 가중치는 거의 균등하게
수렴해도(entropy~0.999), 학습 중 backprop 경로 자체는 무파라미터 MeanPooling과 다르므로
(가중합의 ctx_tokens에 대한 미분 vs 단순평균의 미분은 형태가 다름), self.vit 쪽으로 흘러가는
gradient가 달라져 최종 ctx_tokens의 "질"이 달라질 수 있다.

[검증 방법] attention 학습본과 meanpool 재학습본, 두 체크포인트에서 같은 환자·같은 slide의
ctx_tokens(self.vit 출력, pooling 직전)을 직접 뽑아 slide 내부 patch 간 다양성을 잰다 —
patch 표현들이 서로 얼마나 다른지(낮을수록 다양, 높을수록 collapse/균질)를 평균 pairwise
cosine similarity로 측정(L2-normalize 후 sum-of-squares 항등식으로 O(N) 계산, N이 커도 빠름):
    mean_pairwise_cos = (||sum_i u_i||^2 - N) / (N^2 - N),  u_i = ctx_tokens[i]/||ctx_tokens[i]||

가설이 맞다면(attn_pool의 존재가 self.vit을 더 다양한 표현을 만들게 shaping) attention
학습본의 ctx_tokens가 meanpool 학습본보다 pairwise cosine이 더 낮아야(더 다양해야) 한다.

사용법:
    python -m scripts.diagnose_porpoise_ctxtoken_diversity \
        --ckpt-attn models/checkpoint/survival_..._DISP_FOLD0OF5_best_porpoise.pt \
        --ckpt-meanpool models/checkpoint/survival_..._MEANPOOL_..._best_porpoise.pt \
        --seed 84 --fold 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids_intersection
from models import ViT_PORPOISE
from models.clinical_encoder import age_stats_from_csv, stage_stats_from_df, margin_stats_from_df


@torch.no_grad()
def _ctx_tokens(model, coords, features, device) -> torch.Tensor:
    """models/vit_m1.py::ViT_M1.forward의 patch_tokens -> ctx_tokens 계산을 그대로 재현
    (pooling 직전 중간값만 뽑아냄)."""
    coords = coords.to(device, non_blocking=True)
    features = features.to(device, non_blocking=True) if features is not None else None
    patch_tokens = model._patch_tokens(coords, None, features, None, None, None)
    if model.use_coord_embed:
        coord_input = (coords[torch.randperm(coords.shape[0], device=coords.device)]
                        if model.coord_embed_shuffle else coords)
        pos = model.coord_embed(coord_input)
        if model.coord_embed_concat:
            patch_tokens = model.coord_fusion(torch.cat([patch_tokens, pos], dim=-1))
        elif hasattr(model, "coord_embed_scale"):
            patch_tokens = patch_tokens + model.coord_embed_scale * pos
        else:
            patch_tokens = patch_tokens + pos
    if model.use_wsi_extra_mlp:
        patch_tokens = model.wsi_extra_mlp(patch_tokens)
    ctx_tokens = patch_tokens if model.skip_patch_vit else model.vit(patch_tokens, coords, tumor_type=None)
    return ctx_tokens  # (N_patches, D)


def _mean_pairwise_cosine(tokens: torch.Tensor) -> float:
    """L2-normalize 후 sum-of-squares 항등식으로 O(N) 계산되는 평균 pairwise cosine
    similarity(자기 자신끼리의 쌍은 제외). 1에 가까우면 patch 표현들이 서로 거의 같음(collapse),
    0에 가까우면 서로 다름(다양)."""
    n = tokens.shape[0]
    if n < 2:
        return float("nan")
    u = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-12)
    s = u.sum(dim=0)
    sum_all_incl_self = float((s @ s).item())
    return (sum_all_incl_self - n) / (n * (n - 1))


def _load_model(ckpt_path, cfg, rna_gene_ids, age_mean, age_std, stage_stats, margin_stats,
                 backbone, device, use_meanpool: bool):
    model = ViT_PORPOISE(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(rna_gene_ids),
        backbone=backbone, use_staging=True, stage_stats=stage_stats,
        use_margin=True, margin_stats=margin_stats, use_attn_dispersion=True,
        use_meanpool=use_meanpool,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not missing, f"필수 파라미터 누락: {missing}"
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-attn", type=str, required=True)
    parser.add_argument("--ckpt-meanpool", type=str, required=True)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="tcga")
    parser.add_argument("--rna-n-genes", type=int, default=1500)
    parser.add_argument("--backbone", type=str, default="uni2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.data.seed = args.seed
    rna_gene_ids = literature_guided_gene_ids_intersection(args.rna_n_genes)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset]))
    margin_stats = margin_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset])[["residual_disease"]])
    ds_kwargs = dict(
        with_clinical=True, with_staging=True, with_margin=True, with_rna=True,
        feature_backbone=args.backbone, rna_gene_ids=rna_gene_ids,
        fold=args.fold, n_folds=args.n_folds,
    )
    ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="test", **ds_kwargs)
    print(f"internal({args.dataset} fold{args.fold}/{args.n_folds} test) 환자 수: {len(ds)}")

    model_attn = _load_model(args.ckpt_attn, cfg, rna_gene_ids, age_mean, age_std, stage_stats,
                              margin_stats, args.backbone, device, use_meanpool=False)
    model_mean = _load_model(args.ckpt_meanpool, cfg, rna_gene_ids, age_mean, age_std, stage_stats,
                              margin_stats, args.backbone, device, use_meanpool=True)

    div_attn, div_mean = [], []
    for i in range(len(ds)):
        patient_slides = ds[i]
        for slide in patient_slides:
            coords = slide["coords"]
            features = slide.get("features")
            ct_attn = _ctx_tokens(model_attn, coords, features, device)
            ct_mean = _ctx_tokens(model_mean, coords, features, device)
            div_attn.append(_mean_pairwise_cosine(ct_attn))
            div_mean.append(_mean_pairwise_cosine(ct_mean))

    div_attn = np.array(div_attn)
    div_mean = np.array(div_mean)
    print(f"\nslide 수: {len(div_attn)}")
    print(f"[attention 학습본] mean pairwise cosine(patch 간) = {np.nanmean(div_attn):.4f} +/- {np.nanstd(div_attn):.4f}")
    print(f"[meanpool  학습본] mean pairwise cosine(patch 간) = {np.nanmean(div_mean):.4f} +/- {np.nanstd(div_mean):.4f}")
    diff = div_attn - div_mean
    print(f"차이(attn - meanpool): mean={np.nanmean(diff):+.4f}, "
          f"음수면 attention 학습본 ctx_tokens가 더 다양(덜 collapse) — 가설과 일치")
    # paired t-test (slide 단위)
    from scipy import stats as sstats
    valid = ~(np.isnan(diff))
    t, p = sstats.ttest_rel(div_attn[valid], div_mean[valid])
    print(f"paired t-test: t={t:.3f}, p={p:.4f}")


if __name__ == "__main__":
    main()
