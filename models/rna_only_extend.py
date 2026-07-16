"""
RNAOnlyExtend — M6X, RNAEncoderExtend(레퍼런스 사양: G -> 256 -> 256, dropout 0.25)를 쓰는
RNAOnly(M6)의 변형. train.py --M6X.

목적: "RNA 브랜치를 강화하면 도움이 될까?"라는 질문을 유전자 재선정(입력 차원 확대,
아직 미구현)과 분리해서 먼저 검증한다 — 입력 유전자는 M6와 동일하게 339개(Bailey/Moffitt
subtype)로 고정하고, 인코더 폭/dropout만 레퍼런스 M3/M4 RNA branch 사양으로 바꿔 그 자체의
효과만 통제 비교한다. WSI가 없는 가장 가벼운 모델이라 이 실험을 가장 먼저 돌린다.
"""
import torch
import torch.nn as nn

from .rna_encoder_extend import RNAEncoderExtend
from config import ModelConfig


class RNAOnlyExtend(nn.Module):
    def __init__(self, cfg: ModelConfig, rna_input_dim: int, rna_embed_dim: int = 256):
        super().__init__()
        self.rna_encoder = RNAEncoderExtend(rna_input_dim, embed_dim=rna_embed_dim, dropout=0.25)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(rna_embed_dim),
            nn.Linear(rna_embed_dim, 1),
        )

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rna: (G,) — 코호트 내부 z-score 정규화된 유전자 발현 벡터
        Returns:
            risk: (1,)
        """
        z = self.rna_encoder(rna.unsqueeze(0)).squeeze(0)  # (E,) 기본 E=256
        return self.risk_head(z.unsqueeze(0)).view(1)
