"""
scripts/diagnose_nystrom_oversmoothing.py의 PAAD(TCGA) 버전. BRCA(패치 수 중앙값 8500~10000,
num_landmarks=128보다 훨씬 많음)에서는 Nystrom 통과 후 oversmoothing 징후가 없었다(cosine
similarity 거의 안 변함, effective rank 오히려 소폭 상승) — 그런데 애초에 ABMIL attention
entropy~0.999 붕괴가 실측된 건 PAAD였다(2026-08-14, diagnose_pma_wsi_structure.py, 패치 수
중앙값 67 << num_landmarks 128). patches 수가 landmark 수보다 훨씬 적은 이 영역에서 Nystrom의
landmark 근사가 실제로는 "거의 모든 패치를 landmark로 쓰는 셈"이 되어 dense self-attention에
가깝게 동작한다 — BRCA와는 근본적으로 다른 영역이라 여기서 직접 재측정해야 진짜 비교가 된다.

사용법:
    python -m scripts.diagnose_nystrom_oversmoothing_paad --ckpt <path> --backbone uni2native
"""
import argparse
import sys
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
from train import _identity_collate


def _mean_pairwise_cosine(tokens: torch.Tensor) -> float:
    n = tokens.shape[0]
    if n < 2:
        return float("nan")
    x = torch.nn.functional.normalize(tokens.float(), dim=-1)
    s = x.sum(dim=0)
    total = (s @ s) - n
    return (total / (n * (n - 1))).item()


def _effective_rank(tokens: torch.Tensor) -> float:
    x = tokens.float()
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    p = eigvals / eigvals.sum().clamp(min=1e-12)
    p = p[p > 1e-12]
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="uni2native")
    parser.add_argument("--num-transformer-layers", type=int, required=True)
    parser.add_argument("--train-dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument("--max-patients", type=int, default=60)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.model.num_transformer_layers = args.num_transformer_layers

    rna_gene_ids = pdac_consistency_gene_ids(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.train_dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.train_dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.train_dataset]))

    ds_kwargs = dict(
        with_clinical=True, with_staging=True, with_margin=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone=args.backbone,
    )
    ds = WSISurvivalDataset(cfg.data, dataset=args.train_dataset, split="all", **ds_kwargs)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    print(f"{args.train_dataset} N={len(ds)}, backbone={args.backbone}")

    rna_input_dim = len(rna_gene_ids) + 8
    model = ViT_PMA(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=cfg.data.precomputed, backbone=args.backbone, combine_mode="cox_add",
        use_staging=True, stage_stats=stage_stats, use_margin=True, margin_stats=margin_stats,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"체크포인트 로드: {Path(args.ckpt).name} (epoch={ckpt.get('epoch')}, val_c_index={ckpt.get('val_c_index')})")
    print(f"num_transformer_layers={cfg.model.num_transformer_layers}")

    captured = {}

    def _hook(module, inputs, output):
        captured["before"] = inputs[0].detach()
        captured["after"] = output.detach()

    handle = model.vit.register_forward_hook(_hook)

    rows = []
    with torch.no_grad():
        for i, patient_slides in enumerate(loader):
            if not patient_slides or i >= args.max_patients:
                if i >= args.max_patients:
                    break
                continue
            all_feats = torch.cat([s["features"] for s in patient_slides], dim=0).to(device).float()
            patch_tokens = model.cnn.forward_pooled(all_feats) if hasattr(model.cnn, "forward_pooled") else model.cnn(all_feats)
            coords = None
            if "coords" in patient_slides[0]:
                coords = torch.cat([s["coords"] for s in patient_slides], dim=0).to(device)
            _ = model.vit(patch_tokens, coords)
            n = captured["before"].shape[0]
            rows.append({
                "n_patches": n,
                "cos_before": _mean_pairwise_cosine(captured["before"]),
                "cos_after": _mean_pairwise_cosine(captured["after"]),
                "rank_before": _effective_rank(captured["before"]),
                "rank_after": _effective_rank(captured["after"]),
            })
    handle.remove()

    df = pd.DataFrame(rows)
    print(f"\n=== 환자 {len(df)}명 평균 (embed_dim={cfg.model.embed_dim}, num_transformer_layers={cfg.model.num_transformer_layers}) ===")
    print(f"패치 수: mean={df['n_patches'].mean():.0f} (median={df['n_patches'].median():.0f})")
    print(f"평균 pairwise cosine similarity: BEFORE={df['cos_before'].mean():.4f} -> AFTER={df['cos_after'].mean():.4f} "
          f"(증가폭={df['cos_after'].mean() - df['cos_before'].mean():+.4f})")
    print(f"effective rank (D={cfg.model.embed_dim}가 최댓값): BEFORE={df['rank_before'].mean():.2f} -> AFTER={df['rank_after'].mean():.2f} "
          f"(감소폭={df['rank_after'].mean() - df['rank_before'].mean():+.2f})")


if __name__ == "__main__":
    main()
