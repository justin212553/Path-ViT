"""
ClinicalOnly — M5, Clinical(age/sex)만 사용하는 WSI/RNA-free baseline. train.py --M5.

train_clinical_rna_only.py::ClinicalRNAOnly(M7, Clinical+RNA 결합)에서 RNA 브랜치를 뺀
절반 버전 — "clinical 정보 단독으로 얼마나 예측되는가"를 보여주는 구색용 ablation이다.
M7과 달리 별도 스크립트가 아니라 train.py에 배선해 --dataset/--seed/--external/wandb
로깅을 다른 M-계열과 동일하게 공유한다.
"""
import torch
import torch.nn as nn

from .clinical_encoder import ClinicalEncoder
from config import ModelConfig


class ClinicalOnly(nn.Module):
    def __init__(self, cfg: ModelConfig, age_mean: float, age_std: float):
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(cfg.embed_dim, age_mean, age_std)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, 1),
        )

    def forward(self, age_years: torch.Tensor, sex_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            age_years: () — 환자 나이(연 단위) 스칼라 텐서
            sex_idx:   () — encode_sex() 인덱스 스칼라 텐서 (0=male, 1=female)
        Returns:
            risk: (1,)
        """
        z = self.clinical_encoder(age_years.unsqueeze(0), sex_idx.unsqueeze(0)).squeeze(0)  # (D,)
        return self.risk_head(z.unsqueeze(0)).view(1)
