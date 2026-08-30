"""
ClinicalOnly — M5, Clinical(age/sex)만 사용하는 WSI/RNA-free baseline. train_light.py --M5.

train_clinical_rna_only.py::ClinicalRNAOnly(M7, Clinical+RNA 결합)에서 RNA 브랜치를 뺀
절반 버전 — "clinical 정보 단독으로 얼마나 예측되는가"를 보여주는 구색용 ablation이다.

2026-08-21: raw_linear 옵션을 추가하고 최종(default 채택)으로 확정 — M2/M4/M7의 clinical
cox_add가 전부 raw feature 직결(ClinicalEncoder(MLP) 없이 z-score feature를 바로 risk
스칼라에 사용, models/vit_pma.py·models/clinical_rna_only.py 참조)로 원복됐는데, M5만
ClinicalEncoder(MLP)를 쓸 architectural 근거가 없다는 지적(사용자) — 실측해봐도 raw_linear가
MLP 버전과 오차범위 내(internal -0.006, external -0.019)라 통일하는 쪽을 택함. 이제 clinical
브랜치는 전 모델(M2/M4/M5/M7)에서 예외 없이 raw feature 직결 — 학습되는 nonlinear 인코더는
RNA/WSI 브랜치(RNAEncoder/ViT)에만 쓴다는 아키텍처 원칙이 완성됨. MLP 버전(raw_linear=False,
과거 M5 확정치 internal=0.5536/external=0.5511)은 참고용 비교 자료로 결과표 부록에 보존.
"""
import torch
import torch.nn as nn

from .clinical_encoder import ClinicalEncoder, STAGE_FIELDS, _STAGE_BUFFER_NAMES
from config import ModelConfig


class ClinicalOnly(nn.Module):
    def __init__(
        self, cfg: ModelConfig, age_mean: float, age_std: float,
        use_staging: bool = False, stage_stats: dict[str, tuple[float, float]] | None = None,
        use_margin: bool = False, margin_stats: tuple[float, float] | None = None,
        use_age_sex: bool = True,
        raw_linear: bool = False,
    ):
        super().__init__()
        self.raw_linear = raw_linear
        self.use_staging = use_staging
        self.use_margin = use_margin
        self.use_age_sex = use_age_sex
        if raw_linear:
            # ClinicalEncoder(MLP) 없이 raw z-score feature -> Linear(1) 직결(고전적 Cox 회귀).
            # models/clinical_rna_only.py::ClinicalRNAOnly의 cox_add raw feature 계산과 동일 관례.
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
            self.risk_head = nn.Linear(raw_dim, 1)
        else:
            self.clinical_encoder = ClinicalEncoder(
                cfg.embed_dim, age_mean, age_std, use_staging=use_staging, stage_stats=stage_stats,
                use_margin=use_margin, margin_stats=margin_stats, use_age_sex=use_age_sex,
            )
            self.risk_head = nn.Sequential(
                nn.LayerNorm(cfg.embed_dim),
                nn.Linear(cfg.embed_dim, 1),
            )

    def _raw_feat(self, age_years: torch.Tensor, sex_idx: torch.Tensor,
                  margin_ord: torch.Tensor | None = None,
                  stage_ord: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """raw_linear=True 전용 — (raw_dim,) z-score feature 반환."""
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
        return torch.stack(feats, dim=-1)  # (raw_dim,)

    def forward(
        self, age_years: torch.Tensor, sex_idx: torch.Tensor,
        stage_ord: dict[str, torch.Tensor] | None = None,
        margin_ord: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            age_years:  () — 환자 나이(연 단위) 스칼라 텐서
            sex_idx:    () — encode_sex() 인덱스 스칼라 텐서 (0=male, 1=female)
            stage_ord:  self.use_staging=True(--clinical-staging)일 때만 필요.
                        {field: () 스칼라 long} — encode_stage_value() 규약.
            margin_ord: self.use_margin=True(--clinical-margin, M5_R)일 때만
                        필요. () 스칼라 long — encode_margin_value() 규약.
        Returns:
            risk: (1,)
        """
        if self.raw_linear:
            feats = self._raw_feat(age_years, sex_idx, margin_ord, stage_ord)  # (raw_dim,)
            return self.risk_head(feats.unsqueeze(0)).view(1)
        extra_kwargs = {}
        if stage_ord is not None:
            extra_kwargs["stage_ord"] = {k: v.unsqueeze(0) for k, v in stage_ord.items()}
        if margin_ord is not None:
            extra_kwargs["margin_ord"] = margin_ord.unsqueeze(0)
        z = self.clinical_encoder(
            age_years.unsqueeze(0), sex_idx.unsqueeze(0), **extra_kwargs
        ).squeeze(0)  # (D,)
        return self.risk_head(z.unsqueeze(0)).view(1)
