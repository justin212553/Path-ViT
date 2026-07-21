"""
RNAEncoderExtend — 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology) `scripts/models/
tabular_survival.py::RNASeqEmbedding`을 그대로 이식한 인코더(2026-07-19, GitHub 원문
직접 확인).

rna_encoder.py::RNAEncoder와의 차이:
  - **입력 정규화**: LayerNorm(G) + Dropout을 첫 Linear *이전*에 건다 — 원본 z-score
    유전자 발현을 다시 한번 인코더 자체 스케일로 정규화하고, 입력 단계에서부터 dropout으로
    노이즈를 준다(우리 기존 RNAEncoder는 입력을 그대로 첫 Linear에 넣음).
  - **출력 정규화**: 마지막 Linear 뒤에도 LayerNorm + GELU를 한 번 더 건다(우리 기존
    RNAEncoder는 마지막 Linear에서 바로 끝남 - 사실상 활성화 없는 선형 출력).
  - embed_dim: 256(기존 RNAEncoder는 cfg.embed_dim=64로 압축) — 레퍼런스는 RNA 임베딩을
    병리/clinical 임베딩과 같은 차원으로 억지로 맞추지 않고 넓게 유지한다.
  - dropout: 0.25(기존 RNAEncoder는 0.3), 단 위치가 다르다(입력+은닉 2곳에 각각 적용).
"""
import torch
import torch.nn as nn


class RNAEncoderExtend(nn.Module):
    """
    gene expression (G,) → 임베딩 (E,). 레퍼런스 RNASeqEmbedding과 동일한 구조:
    LayerNorm(G) -> Dropout -> Linear(G->H) -> GELU -> Dropout -> Linear(H->E) -> LayerNorm(E) -> GELU.
    """

    def __init__(self, input_dim: int, embed_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rna: (N, G) — 코호트 내부 z-score 정규화된 유전자 발현
        Returns:
            z_rna: (N, E) — 기본 E=256
        """
        return self.mlp(rna)
