"""
ClinicalRNAFusionViT — ViT+ABMIL(WSI) + Clinical(age/sex) MLP + RNA-seq MLP Late Fusion 모델

train.py의 --M4 플래그로 선택되는 3-모달 모델. patch_vit_clinical_fusion.py::ClinicalFusionViT
(WSI+Clinical, 2D)와 같은 구조를 그대로 확장해 RNA-seq 임베딩(rna_encoder.py::RNAEncoder)을
세 번째 모달리티로 추가한다.

clinical/RNA 정보 모두 슬라이드가 아니라 환자(case) 단위 메타데이터이므로, forward()가
아니라 환자 단위로 WSI 임베딩을 평균 풀링한 뒤 combine_with_clinical_rna()로 결합한다
(train.py::_patient_risk 참조).
"""
import torch
import torch.nn as nn

from .patch_vit import PatchViT
from .clinical_encoder import ClinicalEncoder
from .rna_encoder import RNAEncoder
from config import ModelConfig


class ClinicalRNAFusionViT(PatchViT):
    """
    ViT+ABMIL(WSI 임베딩) + Clinical MLP(age/sex 임베딩) + RNA-seq MLP(유전자 발현 임베딩)
    Late Fusion. cnn/vit/attn_pool과 슬라이드 단위 forward()는 PatchViT를 그대로 물려받는다.

    [Fusion 구조]
      z_wsi      (D,) — 환자 단위로 평균 풀링된 WSI 임베딩 (train.py에서 계산)
      z_clinical (D,) — ClinicalEncoder(age_years, sex_idx) 출력
      z_rna      (D,) — RNAEncoder(gene_expression) 출력
        → combine_with_clinical_rna()에서 concat → (3D,) → LayerNorm → Linear → risk_score (1,)
    """

    def __init__(
        self,
        cfg: ModelConfig,
        age_mean: float,
        age_std: float,
        rna_input_dim: int,
        precomputed: bool = True,
    ):
        super().__init__(cfg, precomputed)
        self.clinical_encoder = ClinicalEncoder(cfg.embed_dim, age_mean, age_std)
        self.rna_encoder = RNAEncoder(rna_input_dim, cfg.embed_dim, dropout=cfg.dropout)

        # Late Fusion risk head: [z_wsi ‖ z_clinical ‖ z_rna] (3D,) → risk_score (1,)
        # PatchViT가 만든 D 차원 risk_head를 3D 차원으로 교체한다.
        self.risk_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim * 3),
            nn.Linear(cfg.embed_dim * 3, 1),
        )

    def combine_with_clinical_rna(
        self,
        patient_embed: torch.Tensor,
        age_years: torch.Tensor,
        sex_idx: torch.Tensor,
        rna: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            patient_embed: (D,) — 환자 단위로 평균 풀링된 WSI 임베딩
            age_years:     ()   — 환자 나이(연 단위) 스칼라 텐서
            sex_idx:       ()   — encode_sex() 인덱스 스칼라 텐서 (0=male, 1=female)
            rna:           (G,) — 코호트 내부 z-score 정규화된 유전자 발현 벡터
        Returns:
            fused: (3D,) — risk_head 입력
        """
        z_clinical = self.clinical_encoder(age_years.unsqueeze(0), sex_idx.unsqueeze(0)).squeeze(0)  # (D,)
        z_rna      = self.rna_encoder(rna.unsqueeze(0)).squeeze(0)                                    # (D,)
        return torch.cat([patient_embed, z_clinical, z_rna], dim=-1)  # (3D,)
