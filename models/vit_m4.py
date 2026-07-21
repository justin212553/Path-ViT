"""
ViT_M4 — ViT+ABMIL(WSI) + Clinical(age/sex) MLP + RNA-seq MLP Late Fusion 모델
         + RNA-guided attention pooling (Leeyoungsup/pancreatic_cancer_pathology
         scripts/models/m3_pathology_rnaseq_mil.py::RNASeqGuidedPathologyFusion에서
         "post-hoc 아핀 게이트" 대신 "attention pooling 자체를 RNA로 조건화"하는 방식으로 변형)

train.py의 --M4 플래그로 선택되는 3-모달 모델. vit_m2.py::ViT_M2(WSI+Clinical, 2D)와
같은 구조를 그대로 확장해 RNA-seq 임베딩(rna_encoder.py::RNAEncoder)을 세 번째
모달리티로 추가한다.

clinical/RNA 정보 모두 슬라이드가 아니라 환자(case) 단위 메타데이터이므로, forward()가
아니라 환자 단위로 WSI 임베딩을 평균 풀링한 뒤 combine_with_clinical_rna()로 결합한다
(train.py::_patient_risk 참조). 다만 RNA만은 예외적으로 encode_rna()를 슬라이드 루프
*이전*에 호출해, 각 슬라이드의 attn_pool(ABMIL)에 rna_context로 전달해야 한다
(아래 "RNA-guided attention pooling" 설명 참조).

[RNA-guided attention pooling을 쓰는 이유]
단순 concat(z_wsi ‖ z_clinical ‖ z_rna)이나, WSI 임베딩을 다 만든 뒤 RNA로 게이팅하는
post-hoc 아핀변환(z_wsi_gated = z_wsi * sigmoid(W·z_rna))은 risk_head 또는 게이트
단계에서만 두 모달리티가 상호작용해, "RNA subtype에 따라 어떤 패치(형태학적 영역)를
더 볼지"는 학습할 수 없다 — ABMIL이 patch attention을 이미 RNA와 무관하게 결정해버린
뒤이기 때문이다. 대신 ViT_M1::AttentionPooling의 gated-attention 게이트(tanh·sigmoid)에
z_rna를 FiLM식 additive bias로 더해(context_dim), attention *score 계산 자체*를 RNA로
조건화한다 — genomic-guided co-attention MIL(MCAT 계열)과 같은 방향. patient_embed는
이미 RNA-informed 상태로 나오므로, combine_with_clinical_rna()에서는 별도 게이트 없이
[z_wsi ‖ z_clinical ‖ z_rna]만 concat한다.
"""
import torch
import torch.nn as nn

from .vit_m1 import ViT_M1, AttentionPooling
from .clinical_encoder import ClinicalEncoder
from .rna_encoder import RNAEncoder
from config import ModelConfig


class ViT_M4(ViT_M1):
    """
    ViT+ABMIL(WSI 임베딩, RNA-guided) + Clinical MLP(age/sex 임베딩) + RNA-seq MLP(유전자
    발현 임베딩) Late Fusion. cnn/vit는 ViT_M1을 그대로 물려받지만, attn_pool은 RNA
    컨텍스트를 받을 수 있도록 context_dim이 있는 버전으로 교체한다(아래 __init__ 참조).

    [Fusion 구조]
      z_rna        (D,) — RNAEncoder(gene_expression) 출력. encode_rna()로 슬라이드 루프
                           이전에 미리 계산해 각 슬라이드 forward(rna_context=z_rna)에 전달
      z_wsi        (D,) — attn_pool이 z_rna로 조건화된 상태로 슬라이드별 집계 후 환자 단위
                           평균 풀링된 WSI 임베딩 (train.py에서 계산)
      z_clinical   (D,) — ClinicalEncoder(age_years, sex_idx) 출력
        → combine_with_clinical_rna()에서 [z_wsi ‖ z_clinical ‖ z_rna] concat
          → (3D,) → LayerNorm → Linear → risk_score (1,)
    """

    def __init__(
        self,
        cfg: ModelConfig,
        age_mean: float,
        age_std: float,
        rna_input_dim: int,
        precomputed: bool = True,
        backbone: str = "resnet50",
        use_staging: bool = False,
        stage_stats: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__(cfg, precomputed, backbone)
        self.clinical_encoder = ClinicalEncoder(
            cfg.embed_dim, age_mean, age_std, use_staging=use_staging, stage_stats=stage_stats
        )
        self.rna_encoder = RNAEncoder(rna_input_dim, cfg.embed_dim, dropout=cfg.dropout)

        # ViT_M1이 만든 context 없는 attn_pool을, z_rna(D차원)를 attention 게이트에
        # additive bias로 받을 수 있는 버전으로 교체한다 — RNA-guided attention pooling.
        self.attn_pool = AttentionPooling(cfg.embed_dim, context_dim=cfg.embed_dim)

        # Late Fusion risk head: [z_wsi ‖ z_clinical ‖ z_rna] (3D,) → risk_score (1,)
        # ViT_M1이 만든 D 차원 risk_head를 3D 차원으로 교체한다.
        # 2026-07-21: 레퍼런스 M4(m4_pathology_rnaseq_clinical_mil.py::classifier)와 동일하게
        # LayerNorm 뒤 Dropout(0.4) 추가를 시도(은닉층 없이 Dropout만 넣는 최소 개입)했으나
        # negative result(external C 0.614->0.494, findings_backlog.md 13번 항목)로 롤백함 — Cox
        # loss는 배치 내 risk score의 상대적 순서로 손실을 계산해, 최종 스칼라 출력 직전 Dropout이
        # 순서 자체를 크게 흔드는 것으로 추정.
        self.risk_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim * 3),
            nn.Linear(cfg.embed_dim * 3, 1),
        )

    def encode_rna(self, rna: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rna: (G,) — 코호트 내부 z-score 정규화된 유전자 발현 벡터
        Returns:
            z_rna: (D,) — 슬라이드별 forward(rna_context=z_rna)와 combine_with_clinical_rna()
                   양쪽에 전달할 RNA 임베딩. 환자 1명당 한 번만 계산하면 된다
                   (train.py::_patient_risk에서 슬라이드 루프 이전에 호출).
        """
        return self.rna_encoder(rna.unsqueeze(0)).squeeze(0)

    def combine_with_clinical_rna(
        self,
        patient_embed: torch.Tensor,
        age_years: torch.Tensor,
        sex_idx: torch.Tensor,
        z_rna: torch.Tensor,
        stage_ord: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            patient_embed: (D,) — 환자 단위로 평균 풀링된 WSI 임베딩 (attn_pool이 이미
                           z_rna로 조건화되어 RNA-informed 상태)
            age_years:     ()   — 환자 나이(연 단위) 스칼라 텐서
            sex_idx:       ()   — encode_sex() 인덱스 스칼라 텐서 (0=male, 1=female)
            z_rna:         (D,) — encode_rna()로 미리 계산한 RNA 임베딩(슬라이드 루프와 공유)
            stage_ord:     self.clinical_encoder.use_staging=True(--clinical-staging)일 때만
                           필요. {field: () 스칼라 long} — encode_stage_value() 규약.
        Returns:
            fused: (3D,) — risk_head 입력
        """
        stage_kwargs = {}
        if stage_ord is not None:
            stage_kwargs["stage_ord"] = {k: v.unsqueeze(0) for k, v in stage_ord.items()}
        z_clinical = self.clinical_encoder(
            age_years.unsqueeze(0), sex_idx.unsqueeze(0), **stage_kwargs
        ).squeeze(0)  # (D,)
        return torch.cat([patient_embed, z_clinical, z_rna], dim=-1)  # (3D,)
