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

    mode="snn": 2026-08-13, PORPOISE(Chen et al. 2022)의 SNN_Block을 그대로 이식한 대안
    (LayerNorm 제거, GELU+Dropout 대신 ELU+AlphaDropout). AlphaDropout은 0으로 마스킹하는
    대신 SELU의 음수 포화값으로 마스킹+스케일해 평균/분산을 유지한다 — 표준 Dropout보다
    표현을 덜 파괴적으로 흔들면서 정규화한다는 보고가 있어, 반복적으로 관측된 RNA 브랜치
    과적합(train-val c_index 격차)을 줄이는지 파일럿으로 확인한다. PORPOISE 원본은
    nn.SELU()가 아니라 nn.ELU()를 쓴다(엄밀한 self-normalizing 고정점 보장은 없는 실용적
    변형) — 그대로 재현.
    """

    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.3,
                 mode: str = "gelu"):
        super().__init__()
        if mode == "gelu":
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
            )
        elif mode == "snn":
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ELU(),
                nn.AlphaDropout(p=dropout),
                nn.Linear(hidden_dim, embed_dim),
            )
        else:
            raise ValueError(f"mode must be 'gelu' or 'snn', got {mode!r}")

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rna: (N, G) — 코호트 내부 z-score 정규화된 유전자 발현
        Returns:
            z_rna: (N, D)
        """
        return self.mlp(rna)
