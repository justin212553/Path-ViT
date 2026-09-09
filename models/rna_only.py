"""
RNAOnly — M6, RNA-seq만 사용하는 WSI/Clinical-free baseline. train.py --M6.

ClinicalOnly(M5)의 대칭 버전 — "RNA 발현 프로파일 단독으로 얼마나 예측되는가"를 보여주는
구색용 ablation이다. M7(train_clinical_rna_only.py, Clinical+RNA 결합)이 M5/M6 각각의
단일 모달리티보다 얼마나 더 나은지 비교하는 기준선 역할을 한다.
"""
import torch
import torch.nn as nn

from .rna_encoder import RNAEncoder
from config import ModelConfig


class RNAOnly(nn.Module):
    def __init__(self, cfg: ModelConfig, rna_input_dim: int, rna_encoder_mode: str = "gelu",
                 surv_n_classes: int = 1):
        super().__init__()
        self.rna_encoder = RNAEncoder(rna_input_dim, cfg.embed_dim, dropout=cfg.dropout, mode=rna_encoder_mode)
        # surv_n_classes>1: train_light.py --surv-loss nll_surv/both 전용(models/clinical_rna_only.py::
        # ClinicalRNAOnly와 동일 관례, 2026-09-09 이식). 기본값 1이면 기존 Cox 레시피와 동일.
        self.surv_n_classes = surv_n_classes
        self.risk_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, surv_n_classes),
        )

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rna: (G,) — 코호트 내부 z-score 정규화된 유전자 발현 벡터
        Returns:
            risk: (1,) — surv_n_classes=1(기본). surv_n_classes>1이면 (surv_n_classes,) hazard logits.
        """
        z = self.rna_encoder(rna.unsqueeze(0)).squeeze(0)  # (D,)
        return self.risk_head(z.unsqueeze(0)).view(-1)
