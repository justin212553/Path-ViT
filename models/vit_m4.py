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
from .clinical_encoder import ClinicalEncoder, STAGE_FIELDS, _STAGE_BUFFER_NAMES
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
        use_margin: bool = False,
        margin_stats: tuple[float, float] | None = None,
        use_age_sex: bool = True,
        combine_mode: str = "concat",
        use_attn_dispersion: bool = False,
        skip_patch_vit: bool = False,
        use_clinical: bool = True,
        use_coord_embed: bool = False,
        coord_embed_concat: bool = False,
        coord_embed_learnable_scale: bool = False,
        coord_embed_shuffle: bool = False,
        use_wsi_extra_mlp: bool = False,
    ):
        super().__init__(cfg, precomputed, backbone, use_attn_dispersion=use_attn_dispersion,
                          skip_patch_vit=skip_patch_vit, use_coord_embed=use_coord_embed,
                          coord_embed_concat=coord_embed_concat,
                          coord_embed_learnable_scale=coord_embed_learnable_scale,
                          coord_embed_shuffle=coord_embed_shuffle,
                          use_wsi_extra_mlp=use_wsi_extra_mlp)
        if combine_mode not in ("concat", "cox_add"):
            raise ValueError(f"알 수 없는 combine_mode: {combine_mode}")
        self.combine_mode = combine_mode
        self.use_staging = use_staging
        self.use_margin = use_margin
        self.use_age_sex = use_age_sex
        # 2026-08-14: M3(WSI+RNA, clinical 제외) 슬롯을 M4-NOVIT과 같은 계열(단일 gated ABMIL,
        # RNA-guided FiLM, patch-mixing 없음)로 다시 만들기 위한 옵션 — models/vit_pma.py::
        # ViT_PMA의 use_clinical과 동일 관례. False면 clinical_encoder/clinical_linear를 아예
        # 안 만들고 risk_head 입력이 [z_wsi, z_rna]만으로 구성된다(train.py --no-clinical).
        self.use_clinical = use_clinical
        self.rna_encoder = RNAEncoder(rna_input_dim, cfg.embed_dim, dropout=cfg.dropout)

        # ViT_M1이 만든 context 없는 attn_pool을, z_rna(D차원)를 attention 게이트에
        # additive bias로 받을 수 있는 버전으로 교체한다 — RNA-guided attention pooling.
        self.attn_pool = AttentionPooling(cfg.embed_dim, context_dim=cfg.embed_dim)

        # 2026-08-11: models/vit_pma.py::ViT_PMA/vit_m2.py::ViT_M2와 동일한 관례 — cox_add면
        # clinical을 임베딩해 concat하지 않고 risk_head 스칼라에 고전적 Cox 가산항으로 직접
        # 더한다(train.py::_patient_risk 공용 dispatch가 model.combine_mode를 보고 처리).
        # M4A(patch-level co-attention, MCAT 스타일)를 지금의 최종 레시피(margin/staging/
        # cox_add/attn-dispersion)와 공정하게 비교하기 위해 이식 — 기존엔 M4/M4A가 이 세
        # 플래그를 아예 지원하지 않아 findings_backlog.md의 예전 M4A 기록들은 이 레시피
        # 이전 것들이었다.
        if combine_mode == "concat":
            if self.use_clinical:
                self.clinical_encoder = ClinicalEncoder(
                    cfg.embed_dim, age_mean, age_std, use_staging=use_staging, stage_stats=stage_stats,
                    use_margin=use_margin, margin_stats=margin_stats, use_age_sex=use_age_sex,
                )
        else:  # cox_add
            if not self.use_clinical:
                raise ValueError("combine_mode='cox_add'는 use_clinical=True에서만 의미가 있습니다.")
            self.register_buffer("age_mean", torch.tensor(age_mean, dtype=torch.float32))
            self.register_buffer("age_std", torch.tensor(age_std, dtype=torch.float32))
            if use_margin:
                m_mean, m_std = margin_stats
                self.register_buffer("margin_mean", torch.tensor(m_mean, dtype=torch.float32))
                self.register_buffer("margin_std", torch.tensor(m_std, dtype=torch.float32))
            if use_staging:
                if stage_stats is None:
                    raise ValueError("use_staging=True면 stage_stats가 필요합니다.")
                for field in STAGE_FIELDS:
                    mean, std = stage_stats[field]
                    short = _STAGE_BUFFER_NAMES[field]
                    self.register_buffer(f"{short}_mean", torch.tensor(mean, dtype=torch.float32))
                    self.register_buffer(f"{short}_std", torch.tensor(std, dtype=torch.float32))
            raw_dim = (2 if use_age_sex else 0) + (2 if use_margin else 0) + (2 * len(STAGE_FIELDS) if use_staging else 0)
            if raw_dim == 0:
                raise ValueError("use_age_sex=False이고 use_margin=False이고 use_staging=False면 clinical 입력이 없습니다.")
            self.clinical_linear = nn.Linear(raw_dim, 1, bias=False)
            nn.init.zeros_(self.clinical_linear.weight)  # 초기엔 clinical 가산항 없는 것과 동일

        # Late Fusion risk head: concat이면 [z_wsi ‖ z_clinical ‖ z_rna] (3D,), cox_add면
        # [z_wsi ‖ z_rna] (2D,) — clinical은 risk_head 밖에서 별도로 더해짐. 둘 다
        # spatial_feat_dim(attn-dispersion, 1) 만큼 추가.
        # 2026-07-21: 레퍼런스 M4(m4_pathology_rnaseq_clinical_mil.py::classifier)와 동일하게
        # LayerNorm 뒤 Dropout(0.4) 추가를 시도(은닉층 없이 Dropout만 넣는 최소 개입)했으나
        # negative result(external C 0.614->0.494, findings_backlog.md 13번 항목)로 롤백함 — Cox
        # loss는 배치 내 risk score의 상대적 순서로 손실을 계산해, 최종 스칼라 출력 직전 Dropout이
        # 순서 자체를 크게 흔드는 것으로 추정.
        spatial_feat_dim = 1 if use_attn_dispersion else 0
        # concat: [z_wsi, z_rna] + (use_clinical=True일 때만 z_clinical) — use_clinical=False면
        # M3(WSI+RNA, clinical 제외) 슬롯이 되어 2D. cox_add는 use_clinical=True 강제(위 guard)라 항상 2D.
        n_components = 2 + (1 if (combine_mode == "concat" and self.use_clinical) else 0)
        risk_input_dim = cfg.embed_dim * n_components + spatial_feat_dim
        self.risk_head = nn.Sequential(
            nn.LayerNorm(risk_input_dim),
            nn.Linear(risk_input_dim, 1),
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
        margin_ord: torch.Tensor | None = None,  # self.use_margin=True일 때만 필요
        spatial_feat: torch.Tensor | None = None,  # (1,) — self.use_attn_dispersion=True일 때만
    ) -> torch.Tensor:
        """
        Args:
            patient_embed: (D,) — 환자 단위로 평균 풀링된 WSI 임베딩 (attn_pool이 이미
                           z_rna로 조건화되어 RNA-informed 상태)
            age_years:     ()   — 환자 나이(연 단위) 스칼라 텐서
            sex_idx:       ()   — encode_sex() 인덱스 스칼라 텐서 (0=male, 1=female)
            z_rna:         (D,) — encode_rna()로 미리 계산한 RNA 임베딩(슬라이드 루프와 공유)
            stage_ord:     self.use_staging=True(--clinical-staging)일 때만 필요.
                           {field: () 스칼라 long} — encode_stage_value() 규약.
            margin_ord:    self.use_margin=True(--clinical-margin)일 때만 필요.
        Returns:
            fused: risk_head 입력. combine_mode="concat": (3D,)(+spatial_feat_dim) —
            [z_wsi ‖ z_clinical ‖ z_rna]. "cox_add": (2D,)(+spatial_feat_dim) — [z_wsi ‖ z_rna],
            clinical은 여기서 안 섞이고 train.py가 _clinical_raw()로 별도 계산해 최종
            스칼라에 더한다(models/vit_pma.py::ViT_PMA와 동일 관례).
        """
        if self.combine_mode == "cox_add" or not self.use_clinical:
            # cox_add는 항상, concat이어도 use_clinical=False(M3: WSI+RNA, clinical 제외)면
            # z_clinical을 아예 안 섞는다 — [z_wsi, z_rna]만.
            fused = torch.cat([patient_embed, z_rna], dim=-1)  # (2D,)
            if spatial_feat is not None:
                fused = torch.cat([fused, spatial_feat], dim=-1)
            return fused
        clinical_kwargs = {}
        if stage_ord is not None:
            clinical_kwargs["stage_ord"] = {k: v.unsqueeze(0) for k, v in stage_ord.items()}
        if margin_ord is not None:
            clinical_kwargs["margin_ord"] = margin_ord.unsqueeze(0)
        z_clinical = self.clinical_encoder(
            age_years.unsqueeze(0), sex_idx.unsqueeze(0), **clinical_kwargs
        ).squeeze(0)  # (D,)
        fused = torch.cat([patient_embed, z_clinical, z_rna], dim=-1)  # (3D,)
        if spatial_feat is not None:
            fused = torch.cat([fused, spatial_feat], dim=-1)
        return fused

    def _clinical_raw(self, age_years: torch.Tensor, sex_idx: torch.Tensor,
                       margin_ord: torch.Tensor | None = None,
                       stage_ord: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """combine_mode="cox_add" 전용 — models/vit_pma.py::ViT_PMA._clinical_raw와 동일 관례.
        stage_ord: self.use_staging=True일 때만 필요. {field: () 스칼라 long} — encode_stage_value() 규약."""
        feats = []
        if self.use_age_sex:
            age_z = (age_years.float() - self.age_mean) / self.age_std
            feats += [age_z, sex_idx.float()]
        if self.use_margin:
            ordv = margin_ord.float()
            known = (ordv >= 0).float()
            z = torch.where(ordv >= 0, (ordv - self.margin_mean) / self.margin_std, torch.zeros_like(ordv))
            feats += [z, known]
        if self.use_staging:
            for field in STAGE_FIELDS:
                short = _STAGE_BUFFER_NAMES[field]
                ordv = stage_ord[field].float()
                known = (ordv >= 0).float()
                mean = getattr(self, f"{short}_mean")
                std = getattr(self, f"{short}_std")
                z = torch.where(ordv >= 0, (ordv - mean) / std, torch.zeros_like(ordv))
                feats += [z, known]
        return torch.stack(feats, dim=-1).unsqueeze(0)  # (1, raw_dim)
