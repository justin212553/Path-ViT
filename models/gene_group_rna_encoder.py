"""
GeneGroupRNAEncoder — RNA를 GeneGroupEncoder(8개 PDAC 카테고리, 카테고리마다 독립된 학습되는
Linear 결합)로 인코딩하고, CNV(있으면)를 9번째 "카테고리"로 취급해 같은 방식(LayerNorm+Linear)
으로 인코딩한 뒤, 전부 concat해서 하나의 벡터로 재투영한다 — 기존 RNAEncoder(rna_encoder.py,
flat concat → MLP)를 완전히 대체하는 드롭인 모듈(vit_m4.py::ViT_M4.rna_encoder, forward
인터페이스 (B, G[+C]) -> (B, D) 동일).

[배경, 2026-09-06] 이 프로젝트에서 카테고리 집계 실패 원인은 두 가지로 이미 분리 확인됐다:
  1. pathway8(unsigned mean 집계) — external C 0.49~0.52로 실패. 원인은 카테고리 안에서
     방향이 다른 유전자(예: CD8A 좋은 예후 vs FOXP3 나쁜 예후)가 단순 평균으로 상쇄된 것.
  2. MCAT(GeneGroupEncoder의 학습 Linear 집계 + co-attention fusion) — internal C=0.46으로
     더 나쁨. 하지만 진단 결과 카테고리 집계 자체는 정상(환자별로 다른 토큰을 만듦, gradient도
     정상 도달) — 진범은 co-attention이 query 개수와 무관하게 uniform으로 붕괴하는 것이었다.

즉 "카테고리별 학습 Linear 집계"는 이미 검증된 좋은 방법이고, 문제는 항상 그 뒤에 붙은
fusion(co-attention)이었다. 여기서는 그 집계 방법을 그대로 가져오되, fusion을 이 프로젝트의
PORPOISE 레시피가 이미 검증한 BilinearFusion(attention으로 patch를 고르는 데 의존하지 않음)
으로 바꾼다 — 두 개 모듈의 이미 각각 검증된 장점만 새로 조합하는 것.
"""
import torch
import torch.nn as nn

from .gene_group_encoder import GeneGroupEncoder


class GeneGroupRNAEncoder(nn.Module):
    """
    Args:
        gene_ids: RNA 입력 벡터 (G,)의 각 위치가 어떤 gene_id인지(CNV 제외, 유전자만).
        gene_sets: {category_name: [gene_id, ...]} — 보통 8개 PDAC 기능별 카테고리.
        cnv_dim: CNV 차원 수(0이면 CNV 없음, 9번째 토큰도 안 만듦).
        token_dim: 카테고리 토큰 1개의 차원(GeneGroupEncoder와 CNV 인코더 공용).
        out_dim: 최종 출력 차원(다른 rna_encoder들과 동일하게 cfg.embed_dim으로 맞춰야
                 BilinearFusion 등 하류가 안 바뀜).

    forward(x: (B, G+cnv_dim)) -> (B, out_dim). B=1만 지원(이 프로젝트의 환자당 batch=1 관례,
    ViT_M4.encode_rna()가 항상 rna.unsqueeze(0) 형태로 호출).
    """

    def __init__(
        self, gene_ids: list[str], gene_sets: dict[str, list[str]], cnv_dim: int,
        token_dim: int, out_dim: int,
    ):
        super().__init__()
        self.gene_group_encoder = GeneGroupEncoder(gene_ids, gene_sets, token_dim)
        self.cnv_dim = cnv_dim
        if cnv_dim > 0:
            self.cnv_encoder = nn.Sequential(
                nn.LayerNorm(cnv_dim),
                nn.Linear(cnv_dim, token_dim, bias=False),
            )
        num_tokens = self.gene_group_encoder.num_groups + (1 if cnv_dim > 0 else 0)
        self.proj = nn.Linear(num_tokens * token_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != 1:
            raise ValueError(f"GeneGroupRNAEncoder는 batch_size=1만 지원(입력 shape={tuple(x.shape)})")
        row = x[0]
        if self.cnv_dim > 0:
            rna, cnv = row[: -self.cnv_dim], row[-self.cnv_dim:]
        else:
            rna, cnv = row, None
        tokens = self.gene_group_encoder(rna)  # (K, token_dim)
        if cnv is not None:
            cnv_token = self.cnv_encoder(cnv.unsqueeze(0))  # (1, token_dim)
            tokens = torch.cat([tokens, cnv_token], dim=0)  # (K+1, token_dim)
        flat = tokens.reshape(1, -1)  # (1, num_tokens*token_dim)
        return self.proj(flat)  # (1, out_dim)
