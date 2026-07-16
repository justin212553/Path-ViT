"""
ClinicalRNAOnly — M7, Clinical(age/sex)+RNA-seq 결합, WSI 없음. train_light.py --M7.

train_clinical_rna_only.py(독립 스크립트)에 있던 모델 정의를 다른 WSI-free 모델(clinical_only.py
::ClinicalOnly, rna_only.py::RNAOnly)과 같은 자리로 옮긴 것 — cfg.model(ModelConfig)을
그대로 받아 embed_dim/dropout을 다른 M5/M6/M6X와 동일하게 맞춘다(아키텍처 폭 차이가 아니라
학습 설정 차이만으로 비교할 수 있게 하기 위함, config.py::LightTrainConfig 참조).
"""
import torch
import torch.nn as nn

from .clinical_encoder import ClinicalEncoder
from .rna_encoder import RNAEncoder
from config import ModelConfig


class ClinicalRNAOnly(nn.Module):
    def __init__(self, cfg: ModelConfig, age_mean: float, age_std: float, rna_input_dim: int):
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(cfg.embed_dim, age_mean, age_std)
        self.rna_encoder = RNAEncoder(rna_input_dim, cfg.embed_dim, dropout=cfg.dropout)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim * 2),
            nn.Linear(cfg.embed_dim * 2, 1),
        )

    def forward(self, age_years: torch.Tensor, sex_idx: torch.Tensor, rna: torch.Tensor) -> torch.Tensor:
        """
        Args:
            age_years: () — 환자 나이(연 단위) 스칼라 텐서
            sex_idx:   () — encode_sex() 인덱스 스칼라 텐서 (0=male, 1=female)
            rna:       (G,) — 코호트 내부 z-score 정규화된 유전자 발현 벡터
        Returns:
            risk: (1,)
        """
        z_c = self.clinical_encoder(age_years.unsqueeze(0), sex_idx.unsqueeze(0)).squeeze(0)  # (D,)
        z_r = self.rna_encoder(rna.unsqueeze(0)).squeeze(0)                                    # (D,)
        fused = torch.cat([z_c, z_r], dim=-1)                                                  # (2D,)
        return self.risk_head(fused.unsqueeze(0)).view(1)
