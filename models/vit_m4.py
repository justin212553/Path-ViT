"""
ViT_M4 — ViT+ABMIL(WSI) + Clinical(age/sex) MLP + RNA-seq MLP Late Fusion 모델
         + RNA-guided sigmoid gating (Leeyoungsup/pancreatic_cancer_pathology
         scripts/models/m3_pathology_rnaseq_mil.py::RNASeqGuidedPathologyFusion 이식)

train.py의 --M4 플래그로 선택되는 3-모달 모델. vit_m2.py::ViT_M2(WSI+Clinical, 2D)와
같은 구조를 그대로 확장해 RNA-seq 임베딩(rna_encoder.py::RNAEncoder)을 세 번째
모달리티로 추가한다.

clinical/RNA 정보 모두 슬라이드가 아니라 환자(case) 단위 메타데이터이므로, forward()가
아니라 환자 단위로 WSI 임베딩을 평균 풀링한 뒤 combine_with_clinical_rna()로 결합한다
(train.py::_patient_risk 참조).

[RNA-guided gating을 추가한 이유]
단순 concat(z_wsi ‖ z_clinical ‖ z_rna)은 risk_head가 세 임베딩의 덧셈적 조합만 학습할 수
있어, "RNA subtype에 따라 병리 표현의 어떤 차원을 강조/억제할지" 같은 곱셈적 상호작용을
표현하지 못한다. RNA 임베딩으로 병리 임베딩의 게이트(0~1)를 만들어 곱한 뒤, 원본 z_wsi와
게이트된 z_wsi_gated를 모두 concat에 남겨 원본 정보 손실 없이 상호작용 항을 추가한다.
게이트는 RNA 임베딩에만 적용되고(LayerNorm→Linear→Sigmoid) 병리 임베딩 자체는 별도
아핀변환 없이 그대로 곱해진다 — 참고 구현과 동일한 비대칭 구조.
"""
import torch
import torch.nn as nn

from .vit_m1 import ViT_M1
from .clinical_encoder import ClinicalEncoder
from .rna_encoder import RNAEncoder
from config import ModelConfig


class ViT_M4(ViT_M1):
    """
    ViT+ABMIL(WSI 임베딩) + Clinical MLP(age/sex 임베딩) + RNA-seq MLP(유전자 발현 임베딩)
    Late Fusion. cnn/vit/attn_pool과 슬라이드 단위 forward()는 ViT_M1을 그대로 물려받는다.

    [Fusion 구조]
      z_wsi        (D,) — 환자 단위로 평균 풀링된 WSI 임베딩 (train.py에서 계산)
      z_clinical   (D,) — ClinicalEncoder(age_years, sex_idx) 출력
      z_rna        (D,) — RNAEncoder(gene_expression) 출력
      gate         (D,) — Sigmoid(Linear(LayerNorm(z_rna))) — RNA가 만드는 병리 게이트
      z_wsi_gated  (D,) — z_wsi * gate — RNA로 재해석된 병리 임베딩
        → combine_with_clinical_rna()에서 [z_wsi ‖ z_wsi_gated ‖ z_clinical ‖ z_rna] concat
          → (4D,) → LayerNorm → Linear → risk_score (1,)
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

        # RNA-guided gate: z_rna → LayerNorm → Linear → Sigmoid → (D,) 게이트.
        # 병리 임베딩(z_wsi) 쪽은 별도 아핀변환 없이 이 게이트와 원소별 곱셈만 적용된다.
        self.rna_gate = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.Sigmoid(),
        )

        # Late Fusion risk head: [z_wsi ‖ z_wsi_gated ‖ z_clinical ‖ z_rna] (4D,) → risk_score (1,)
        # ViT_M1이 만든 D 차원 risk_head를 4D 차원으로 교체한다.
        self.risk_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim * 4),
            nn.Linear(cfg.embed_dim * 4, 1),
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
            fused: (4D,) — risk_head 입력
        """
        z_clinical  = self.clinical_encoder(age_years.unsqueeze(0), sex_idx.unsqueeze(0)).squeeze(0)  # (D,)
        z_rna       = self.rna_encoder(rna.unsqueeze(0)).squeeze(0)                                    # (D,)
        gate        = self.rna_gate(z_rna)                                                             # (D,)
        z_wsi_gated = patient_embed * gate                                                              # (D,)
        return torch.cat([patient_embed, z_wsi_gated, z_clinical, z_rna], dim=-1)  # (4D,)
