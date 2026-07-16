"""
RNAEncoderExtend — 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology) M3/M4의 RNA-seq
branch 사양을 그대로 가져온 확장판 인코더.

rna_encoder.py::RNAEncoder와의 차이(Method.md 4.6절 근거):
  - hidden_dim: 256 (기존과 동일)  →  embed_dim: 256 (기존은 cfg.embed_dim=64로 압축)
    레퍼런스는 f_RNA: G -> 256 -> 256으로, RNA 임베딩을 병리 임베딩 차원에 맞춰 좁히지
    않고 넓게 유지한다("RNA-seq embedding 차원을 256으로 확장").
  - dropout: 0.25 (기존 0.3에서 소폭 완화 — "RNA branch dropout을 0.25로 낮추며")
  - 입력 유전자 수 G: 레퍼런스는 literature-guided seed(PDAC driver/subtype/EMT/stromal/
    immune/proliferation/hypoxia/DNA damage repair 8개 카테고리) + train split 내부
    univariate Cox score test(TCGA/CPTAC 각각) + Stouffer meta-analysis로 순위를 매긴 뒤
    상위 1,000/1,500/2,000개를 사용한다. 이 파일은 인코더 아키텍처만 이식한 것이고,
    해당 유전자 재선정 파이프라인(선정 스크립트)은 아직 별도 작업으로 남아 있다 — 그
    전까지는 기존 pdac_subtype_gene_ids()(339개, Bailey/Moffitt subtype 분류 유전자)를
    G 자리에 그대로 넣어도 동작한다(input_dim만 맞으면 유전자 목록과 무관하게 동작).

[기존 RNAEncoder와의 fusion 호환성 주의]
ViT_M4/M4A/M4B의 combine_with_clinical_rna()는 z_rna가 cfg.embed_dim(기본 64)과 같은
차원이라고 가정하고 WSI/clinical 임베딩과 그대로 concat한다. RNAEncoderExtend는 기본값이
256차원이라 그대로 갈아 끼우면 차원이 안 맞는다 — 실제 M4 계열에 연결하려면 별도 projection
층을 추가하거나 risk_head/attn_pool의 context_dim을 256에 맞춰 조정하는 배선 작업이
추가로 필요하다(이 파일은 아직 그 배선 전, 인코더 자체만 옮겨온 상태).
"""
import torch
import torch.nn as nn


class RNAEncoderExtend(nn.Module):
    """
    gene expression (G,) → 임베딩 (E,) 두 층 MLP. 레퍼런스 M3/M4 RNA branch(f_RNA: G -> 256 -> 256)와
    동일한 폭·dropout을 쓴다 — rna_encoder.py::RNAEncoder가 cfg.embed_dim(64)까지 좁히는 것과 달리,
    RNA 임베딩 자체의 표현력을 유지하기 위해 기본 embed_dim을 256으로 둔다.
    """

    def __init__(self, input_dim: int, embed_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.25):
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
            z_rna: (N, E) — 기본 E=256
        """
        return self.mlp(rna)
