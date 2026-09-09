"""
SelfAttentionPooling — query-free self-attention pooling over a small fixed-size set of
component tokens (K cluster representatives). vit_m4a.py::CoAttentionPooling의 RNA-query
버전과 대칭되는, query 없는 버전 — RNA가 없는 모델(M1/M2)에서 ClusterPool(K개 대표 patch
토큰)을 RNA co-attention 대신 이걸로 집계한다.

[존재 이유, 2026-09-09] M4/PMA 계열은 cluster_pool(models/vit_pma.py)의 K개 성분을
component_coattn(RNA가 query)으로 가중합하지만, M1/M2엔 RNA 자체가 없어 그 쿼리를 만들 수
없다. 표준 Transformer encoder block(self-attention -> FFN, 둘 다 residual+LayerNorm)을
K개 토큰의 시퀀스에 적용해 서로를 참조하게 한 뒤 평균 풀링한다 — "어떤 조직 유형이 중요한가"를
외부 신호 없이 성분들 간의 상호작용만으로 정하는, RNA-guided co-attention의 가장 단순한
대응물이다.
"""
import torch
import torch.nn as nn


class SelfAttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, components: torch.Tensor) -> torch.Tensor:
        """
        Args:
            components: (K, D) — 예: ClusterPool의 K개 군집 대표 토큰.
        Returns:
            pooled: (D,) — self-attention으로 서로를 참조한 뒤 평균 풀링된 단일 임베딩.
        """
        x = components.unsqueeze(0)  # (1, K, D) — nn.MultiheadAttention(batch_first=True) 규약
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(0).mean(dim=0)  # (D,)
