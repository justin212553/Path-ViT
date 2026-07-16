"""
ViT_M4A — Genomic-guided co-attention MIL (MCAT 스타일) ablation, train.py --M4A

vit_m4.py::ViT_M4(FiLM 방식 RNA-guided ABMIL)와 fusion 골격은 완전히 동일하다
(encode_rna() → 슬라이드별 attn_pool(tokens, rna_context) → 환자 단위 평균 풀링 →
combine_with_clinical_rna()에서 [z_wsi ‖ z_clinical ‖ z_rna] concat → risk_head).
attn_pool 하나만 CoAttentionPooling(genomic query → patch cross-attention)으로
바꾼 ablation이라, ViT_M4를 그대로 상속하고 __init__에서 attn_pool만 교체한다.

[ViT_M4 대비 차이 — RNA가 patch attention에 개입하는 방식]
  - ViT_M4(AttentionPooling, context_dim=D): ABMIL 자체 게이트(tanh·sigmoid, 패치
    토큰만으로 계산)에 z_rna를 additive bias로 "더하는" FiLM 방식. 패치 attention의
    1차 결정권은 여전히 패치 토큰 자신에게 있고, RNA는 그걸 살짝 휘게 만드는 역할.
  - ViT_M4A(CoAttentionPooling): z_rna 자체가 query가 되어 패치 토큰(key/value)에 대해
    scaled dot-product cross-attention을 계산한다. "이 RNA subtype과 (임베딩 공간에서)
    가장 유사한 패치가 무엇인가"를 명시적 유사도로 학습하는, RNA가 더 강하게 집계를
    주도하는 방식 — Chen et al.(2021) MCAT의 "genomic token이 query, pathology
    patch가 key/value인 co-attention"을, 유전자를 pathway별로 쪼개지 않고 RNAEncoder가
    만든 단일 (D,) 벡터에 맞게 single-query cross-attention으로 단순화한 버전이다.

    MCAT 원본과 달리 genomic self-attention branch(유전자 자체의 Transformer)는 두지
    않는다 — z_rna가 이미 combine_with_clinical_rna()에서 그대로 concat되므로, "RNA
    자체의 표현"과 "RNA-guided 병리 표현"이 이중으로 들어가지 않게 하기 위함이다.
"""
import torch
import torch.nn as nn

from .vit_m4 import ViT_M4
from config import ModelConfig


class CoAttentionPooling(nn.Module):
    """
    z_rna를 query, ViT를 지난 패치 토큰을 key/value로 쓰는 multi-head cross-attention
    풀링. vit_m1.py::AttentionPooling과 반환 규약(wsi_embed, attn_weights)이 같아
    ViT_M4의 forward()/시각화 코드(save_heatmap)와 그대로 호환된다.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)  # z_rna → query 공간 투영
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self, tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens:  (N, D) — ViT를 지난 패치 토큰
            context: (D,)   — RNA 임베딩(z_rna). CoAttentionPooling은 항상 RNA로만
                     쓰이므로(ViT_M4A는 M4처럼 context 없이 호출되지 않음) 필수 인자다.
        Returns:
            wsi_embed:    (D,) — z_rna가 query로서 패치들을 가중합한 co-attention 결과
            attn_weights: (N,) — query 1개 기준 head-평균 attention 가중치 (합=1)
        """
        query = self.query_proj(context).unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        kv    = tokens.unsqueeze(0)                                  # (1, N, D)
        attn_out, attn_weights = self.mha(
            query, kv, kv, need_weights=True, average_attn_weights=True
        )  # attn_out: (1, 1, D), attn_weights: (1, 1, N)
        wsi_embed = attn_out.squeeze(0).squeeze(0)          # (D,)
        attn_weights = attn_weights.squeeze(0).squeeze(0)   # (N,)
        return wsi_embed, attn_weights


class ViT_M4A(ViT_M4):
    """
    ViT_M4와 동일한 3-모달(WSI+Clinical+RNA) Late Fusion 골격에서, attn_pool만
    RNA-guided co-attention(MCAT 스타일)으로 교체한 ablation.
    encode_rna()/combine_with_clinical_rna()/risk_head(3D)는 ViT_M4 그대로 상속한다.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        age_mean: float,
        age_std: float,
        rna_input_dim: int,
        precomputed: bool = True,
        backbone: str = "resnet50",
        num_heads: int = 4,
    ):
        super().__init__(cfg, age_mean, age_std, rna_input_dim, precomputed, backbone)
        self.attn_pool = CoAttentionPooling(cfg.embed_dim, num_heads=num_heads, dropout=cfg.dropout)
