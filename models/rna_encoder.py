"""
RNAEncoder — RNA-seq 발현 프로파일 MLP 인코더

data/rna_{tcga,cptac}.csv 의 유전자 발현(코호트 내부 z-score 정규화된 TPM,
extract_rna_clinical.py 산출물) 중 Bailey 2016 + Moffitt 2015 PDAC subtype 분류 유전자만
(data/dataset.py::pdac_subtype_gene_ids()) WSI/Clinical 임베딩과 late-fusion할 수 있는
D차원 벡터로 변환한다. clinical_encoder.py::ClinicalEncoder와 같은 역할이지만, 입력이 이미
코호트 내부에서 z-score 정규화되어 있으므로(extract_rna_clinical.py 2절 "데이터셋 내부
z-score 정규화") 별도의 정규화 buffer가 필요 없다.
"""
import torch
import torch.nn as nn


class RNAEncoder(nn.Module):
    """
    gene expression (G,) → 임베딩 (D,) 두 층 MLP.

    입력 차원(G, Bailey+Moffitt subtype 분류 유전자 ~340개)이 case 수(코호트당 ~150)에 비해
    여전히 크므로, hidden_dim을 좁은 병목으로 두고 dropout으로 규제한다.
    """

    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rna: (N, G) — 코호트 내부 z-score 정규화된 유전자 발현
        Returns:
            z_rna: (N, D)
        """
        return self.mlp(rna)
