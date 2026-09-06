"""
사용자 질문(2026-09-05): ABMIL의 gate가 거의 균등분포로 붕괴한 게(2026-08-14 실측,
attention entropy~0.999) embed_dim=64 압축 때문인지, 아니면 self-attention 자체의
oversmoothing(층을 쌓을수록 토큰들이 서로 닮아가는 현상, Vision Transformer/Graph Attention
문헌에서 잘 알려진 rank collapse) 때문인지 확인. "나이스트롬 레이어를 2개로 늘리니 오히려
성능이 더 나빠졌다"는 관찰이 oversmoothing 가설과 맞아떨어진다는 추론을 실측으로 검증한다.

이미 학습된 BRCA PMA 체크포인트(1-layer: survival_brca_best_brca_pma_cons882_stg_ss_aux_
seed84_fold0of5.pt, 2-layer: ..._vitlayers2_es10_seed84_fold0of5.pt)에 forward hook을 걸어
self.vit(ViTEncoder, Nystrom) 직전(patch_tokens, self.cnn 출력 — 이미 64차원으로 압축된 뒤)과
직후(ctx_tokens)의 패치 토큰들을 뽑아, 환자별로:
  1) 평균 pairwise cosine similarity — 토큰들이 서로 얼마나 닮았는지. O(N^2) 대신
     ||sum(x_i)||^2 = sum_i sum_j x_i.x_j 항등식으로 O(N*D)에 계산(N이 BRCA에서 최대
     67,268이라 실제 N^2 행렬을 만들면 메모리/시간이 감당 안 됨).
  2) effective rank(entropy 기반, Roy & Vetterli 2007) — 토큰 행렬의 특이값 분포가 얼마나
     "고르게 퍼져 있는지"(다양한 방향에 정보가 분산되어 있는지, 클수록 좋음)를 D x D
     공분산 행렬의 고유값으로 계산(D=64라 SVD보다 훨씬 저렴).
둘 다 "AFTER(Nystrom 이후)가 BEFORE보다 유사도는 높고 rank는 낮다"면 oversmoothing이 실제로
일어난다는 뜻이고, 1-layer보다 2-layer에서 이 효과가 더 크다면 "레이어를 늘릴수록 나빠진다"는
관찰의 메커니즘이 확인되는 것이다.

사용법:
    python -m scripts.diagnose_nystrom_oversmoothing --ckpt <path> --num-transformer-layers 1
    python -m scripts.diagnose_nystrom_oversmoothing --ckpt <path> --num-transformer-layers 2
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
from models.vit_pma import ViT_PMA
from models.rna_predictor import RNAPredictionHead
from models.clinical_encoder import age_stats_from_csv, stage_stats_from_csv
from scripts.brca_common import (
    CLINICAL_PATH, BRCASlideDataset, load_case_table_kfold, load_rna_matrix, MANIFEST_PATH, EXTERNAL_TSS,
)


def _mean_pairwise_cosine(tokens: torch.Tensor) -> float:
    """tokens: (N, D). O(N*D)로 평균 pairwise cosine similarity 계산."""
    n = tokens.shape[0]
    if n < 2:
        return float("nan")
    x = torch.nn.functional.normalize(tokens.float(), dim=-1)
    s = x.sum(dim=0)
    total = (s @ s) - n  # sum_i sum_j x_i.x_j 중 대각항(=n, 자기 자신과의 내적=1) 제외
    return (total / (n * (n - 1))).item()


def _effective_rank(tokens: torch.Tensor) -> float:
    """tokens: (N, D). D x D 공분산 행렬의 고유값 분포로 entropy 기반 effective rank(Roy &
    Vetterli 2007) 계산 — exp(엔트로피), 최댓값은 D(모든 방향에 고르게 분산), 1에 가까울수록
    사실상 1개 방향(rank-1)으로 붕괴한 것."""
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
    parser.add_argument("--num-transformer-layers", type=int, required=True)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-genes", type=int, default=882)  # consistency 패널 유전자 수(태그 CONS882)
    parser.add_argument("--max-patients", type=int, default=40,
                         help="속도를 위해 test 코호트 중 처음 N명만(기본 40) — patient 순서는 "
                              "고정(shuffle 없음)이라 재현 가능.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.model.num_transformer_layers = args.num_transformer_layers

    gene_path = Path("data/brca_rna_gene_selection_consistency/selected_genes.csv")
    gene_ids = pd.read_csv(gene_path)["gene_id"].tolist()
    rna_input_dim = len(gene_ids)
    stage_stats = stage_stats_from_csv(CLINICAL_PATH)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATH)

    cases = load_case_table_kfold(args.seed, args.fold, args.n_folds, external_tss=None)
    rna_df = load_rna_matrix(gene_ids)
    manifest = pd.read_csv(MANIFEST_PATH)
    test_ds = BRCASlideDataset(cases[cases["split"] == "test"], rna_df, manifest, with_staging=True)
    print(f"test N={len(test_ds)} (--max-patients {args.max_patients}로 제한)")

    model = ViT_PMA(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=True, backbone="uni", use_staging=True, stage_stats=stage_stats,
    ).to(device)
    model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, rna_input_dim).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"체크포인트 로드: {Path(args.ckpt).name} (epoch={ckpt.get('epoch')}, val_c_index={ckpt.get('val_c_index')})")
    print(f"num_transformer_layers={cfg.model.num_transformer_layers}")

    captured = {}

    def _hook(module, inputs, output):
        captured["before"] = inputs[0].detach()   # (N, D) — self.cnn 출력, Nystrom 직전
        captured["after"] = output.detach()        # (N, D) — Nystrom 직후

    handle = model.vit.register_forward_hook(_hook)

    rows = []
    n_patients = min(args.max_patients, len(test_ds))
    with torch.no_grad():
        for i in range(n_patients):
            sample = test_ds[i]
            slides = sample if isinstance(sample, list) else [sample]
            all_feats = torch.cat([s["features"] for s in slides], dim=0).to(device).float()
            patch_tokens = model.cnn.forward_pooled(all_feats) if hasattr(model.cnn, "forward_pooled") else model.cnn(all_feats)
            coords = torch.cat([s["coords"] for s in slides], dim=0).to(device) if "coords" in slides[0] else None
            _ = model.vit(patch_tokens, coords)
            n = captured["before"].shape[0]
            cos_before = _mean_pairwise_cosine(captured["before"])
            cos_after = _mean_pairwise_cosine(captured["after"])
            rank_before = _effective_rank(captured["before"])
            rank_after = _effective_rank(captured["after"])
            rows.append({"n_patches": n, "cos_before": cos_before, "cos_after": cos_after,
                         "rank_before": rank_before, "rank_after": rank_after})
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n_patients} 환자 처리...")
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
