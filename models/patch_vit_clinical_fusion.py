"""
ClinicalFusionViT — ViT+ABMIL(WSI)과 Clinical(age/sex) MLP의 Late Fusion 모델

train.py의 --M2 플래그로 선택되는 멀티모달 모델. 슬라이드 단위로는 PatchViT와 동일하게
patch_tokens → ViT → ABMIL로 WSI 임베딩을 만들고(forward), 환자 단위로 슬라이드 임베딩을
평균 풀링한 뒤(train.py::_patient_risk) combine_with_clinical()로 age/sex 임베딩
(clinical_encoder.py::ClinicalEncoder)을 concat해 risk_head에 넣는다.

clinical 정보는 슬라이드가 아니라 환자(case) 단위 메타데이터이므로, forward()가 아니라
환자 단위 집계 이후 결합한다 — patch_vit_fusion.py::LateFusionViT(Cluster Histogram)이
슬라이드 단위 raw feature로 fusion하는 것과의 차이점이다.
"""
import torch
import torch.nn as nn

from .patch_vit import PatchViT
from .clinical_encoder import ClinicalEncoder
from config import ModelConfig


class ClinicalFusionViT(PatchViT):
    """
    ViT+ABMIL(WSI 임베딩) + Clinical MLP(age/sex 임베딩) Late Fusion.
    cnn/vit/attn_pool과 슬라이드 단위 forward()는 PatchViT를 그대로 물려받는다.

    [Fusion 구조]
      z_wsi      (D,) — 환자 단위로 평균 풀링된 WSI 임베딩 (train.py에서 계산)
      z_clinical (D,) — ClinicalEncoder(age_years, sex_idx) 출력
        → combine_with_clinical()에서 concat → (2D,) → LayerNorm → Linear → risk_score (1,)
    """

    def __init__(
        self,
        cfg: ModelConfig,
        age_mean: float,
        age_std: float,
        precomputed: bool = True,
    ):
        super().__init__(cfg, precomputed)
        self.clinical_encoder = ClinicalEncoder(cfg.embed_dim, age_mean, age_std)

        # Late Fusion risk head: [z_wsi ‖ z_clinical] (2D,) → risk_score (1,)
        # PatchViT가 만든 D 차원 risk_head를 2D 차원으로 교체한다.
        self.risk_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim * 2),
            nn.Linear(cfg.embed_dim * 2, 1),
        )

    def combine_with_clinical(
        self, patient_embed: torch.Tensor, age_years: torch.Tensor, sex_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            patient_embed: (D,) — 환자 단위로 평균 풀링된 WSI 임베딩
            age_years:     ()   — 환자 나이(연 단위) 스칼라 텐서
            sex_idx:       ()   — encode_sex() 인덱스 스칼라 텐서 (0=male, 1=female)
        Returns:
            fused: (2D,) — risk_head 입력
        """
        z_clinical = self.clinical_encoder(age_years.unsqueeze(0), sex_idx.unsqueeze(0)).squeeze(0)  # (D,)
        return torch.cat([patient_embed, z_clinical], dim=-1)  # (2D,)
