"""
ViT_M2 — ViT+ABMIL(WSI)과 Clinical(age/sex) MLP의 Late Fusion 모델

train.py의 --M2 플래그로 선택되는 멀티모달 모델. 슬라이드 단위로는 ViT_M1과 동일하게
patch_tokens → ViT → ABMIL로 WSI 임베딩을 만들고(forward), 환자 단위로 슬라이드 임베딩을
평균 풀링한 뒤(train.py::_patient_risk) combine_with_clinical()로 age/sex 임베딩
(clinical_encoder.py::ClinicalEncoder)을 concat해 risk_head에 넣는다.

clinical 정보는 슬라이드가 아니라 환자(case) 단위 메타데이터이므로, forward()가 아니라
환자 단위 집계 이후 결합한다 — patch_vit_fusion.py::LateFusionViT(Cluster Histogram)이
슬라이드 단위 raw feature로 fusion하는 것과의 차이점이다.
"""
import torch
import torch.nn as nn

from .vit_m1 import ViT_M1
from .clinical_encoder import ClinicalEncoder, STAGE_FIELDS, _STAGE_BUFFER_NAMES
from config import ModelConfig


class ViT_M2(ViT_M1):
    """
    ViT+ABMIL(WSI 임베딩) + Clinical MLP(age/sex 임베딩) Late Fusion.
    cnn/vit/attn_pool과 슬라이드 단위 forward()는 ViT_M1을 그대로 물려받는다.

    [Fusion 구조]
      combine_mode="concat"(기본):
        z_wsi      (D,) — 환자 단위로 평균 풀링된 WSI 임베딩 (train.py에서 계산)
        z_clinical (D,) — ClinicalEncoder(age_years, sex_idx) 출력
          → combine_with_clinical()에서 concat → (2D,) → LayerNorm → Linear → risk_score (1,)
      combine_mode="cox_add"(2026-08-07, models/vit_pma.py::ViT_PMA와 동일 관례):
        clinical(age/sex/staging)을 임베딩해 concat하지 않고, risk_head(z_wsi)의 스칼라 출력에
        고전적 Cox 가산항(clinical_linear, zero-init)을 직접 더한다 — train.py가
        _clinical_raw()로 raw feature를 만들어 별도로 더한다(combine_with_clinical_rna의
        cox_add 분기와 동일 패턴). M2엔 margin 입력이 없어(PMA/ClinicalOnly 전용) age/sex(+staging)
        만 raw feature로 쓴다.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        age_mean: float,
        age_std: float,
        precomputed: bool = True,
        backbone: str = "resnet50",
        use_staging: bool = False,
        stage_stats: dict[str, tuple[float, float]] | None = None,
        use_margin: bool = False,
        margin_stats: tuple[float, float] | None = None,
        use_attn_dispersion: bool = False,
        combine_mode: str = "concat",
        skip_patch_vit: bool = False,
        use_coord_embed: bool = False,
        coord_embed_concat: bool = False,
        coord_embed_learnable_scale: bool = False,
    ):
        super().__init__(cfg, precomputed, backbone, use_attn_dispersion=use_attn_dispersion,
                          skip_patch_vit=skip_patch_vit, use_coord_embed=use_coord_embed,
                          coord_embed_concat=coord_embed_concat,
                          coord_embed_learnable_scale=coord_embed_learnable_scale)
        if combine_mode not in ("concat", "cox_add"):
            raise ValueError(f"알 수 없는 combine_mode: {combine_mode}")
        self.combine_mode = combine_mode
        self.use_staging = use_staging
        # 2026-08-14: M2(WSI+Clinical)를 M4-NOVIT과 같은 계열(단일 ABMIL, patch-mixing 없음)로
        # 맞추면서 margin(절제연) 입력도 함께 지원하게 확장 — 기존엔 M2에 margin이 아예 없었다
        # (PMA/M4/M5/M6/M7만 --clinical-margin 지원). 기본 False라 기존 동작 유지.
        self.use_margin = use_margin

        if combine_mode == "concat":
            self.clinical_encoder = ClinicalEncoder(
                cfg.embed_dim, age_mean, age_std, use_staging=use_staging, stage_stats=stage_stats,
                use_margin=use_margin, margin_stats=margin_stats,
            )
            # Late Fusion risk head: [z_wsi ‖ z_clinical] (2D,) [+ dispersion(1,)] → risk_score (1,)
            risk_input_dim = cfg.embed_dim * 2 + (1 if use_attn_dispersion else 0)
        else:  # cox_add
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
            # age/sex(2) [+ margin(2)] [+ staging(8)]
            raw_dim = 2 + (2 if use_margin else 0) + (2 * len(STAGE_FIELDS) if use_staging else 0)
            self.clinical_linear = nn.Linear(raw_dim, 1, bias=False)
            nn.init.zeros_(self.clinical_linear.weight)  # 초기엔 clinical 없는 M1과 동일
            # cox_add는 risk_head에 clinical을 넣지 않는다 — z_wsi(+dispersion)만.
            risk_input_dim = cfg.embed_dim + (1 if use_attn_dispersion else 0)

        # ViT_M1이 만든 risk_head를 위 차원으로 교체한다.
        self.risk_head = nn.Sequential(
            nn.LayerNorm(risk_input_dim),
            nn.Linear(risk_input_dim, 1),
        )

    def combine_with_clinical(
        self, patient_embed: torch.Tensor, age_years: torch.Tensor, sex_idx: torch.Tensor,
        stage_ord: dict[str, torch.Tensor] | None = None,
        margin_ord: torch.Tensor | None = None,  # self.use_margin=True일 때만 필요
        spatial_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            patient_embed: (D,) — 환자 단위로 평균 풀링된 WSI 임베딩
            age_years:     ()   — 환자 나이(연 단위) 스칼라 텐서
            sex_idx:       ()   — encode_sex() 인덱스 스칼라 텐서 (0=male, 1=female)
            stage_ord:     self.clinical_encoder.use_staging=True(--clinical-staging)일 때만
                           필요. {field: () 스칼라 long} — encode_stage_value() 규약.
            margin_ord:    self.use_margin=True(--clinical-margin)일 때만 필요.
            spatial_feat:  (1,) — self.use_attn_dispersion=True일 때만, 환자 단위 평균
                           dispersion(models/spatial_features.py). 2026-07-30, PMA와 동일 관례.
        Returns:
            fused: risk_head 입력. concat: (2D,)/(2D+1,). cox_add: (D,)/(D+1,) — clinical은
            여기서 섞이지 않고(train.py가 _clinical_raw()로 별도 계산해 최종 스칼라에 더함).
        """
        if self.combine_mode == "cox_add":
            fused = patient_embed
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
        fused = torch.cat([patient_embed, z_clinical], dim=-1)  # (2D,)
        if spatial_feat is not None:
            fused = torch.cat([fused, spatial_feat], dim=-1)
        return fused

    def _clinical_raw(self, age_years: torch.Tensor, sex_idx: torch.Tensor,
                       margin_ord: torch.Tensor | None = None,
                       stage_ord: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """combine_mode="cox_add" 전용 — models/vit_pma.py::ViT_PMA._clinical_raw와 동일 관례."""
        age_z = (age_years.float() - self.age_mean) / self.age_std
        feats = [age_z, sex_idx.float()]
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
