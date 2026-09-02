"""
HDP(Human Doctor Prognosis) — M7(ClinicalRNAOnly, models/clinical_rna_only.py)에 WSI 유래
저차원 구조화 feature 하나를 추가한 모델. train_light.py --HDP.

[배경, 2026-09-01] WSI branch를 처음부터 다시 설계하는 논의 — attention/MIL로 patch 중요도를
152개 생존 라벨만으로 처음부터 발견시키는 접근(이번 세션 내내 MCAT/PORPOISE/sharpening 전부
실패)을 버리고, "patch가 어떤 조직 유형에 속하는지"를 병리 지식으로 미리 구조화한 뒤 넣자는
발상에서 시작했다. 그런데 그 조직 유형에 실제로 이름(종양/기질/...)을 붙이려면 라벨이
필요한데, TCGA/CPTAC 어디에도(clinical.tsv, pathology_detail.tsv 전부) PAAD에 대해 그런
라벨이 없다는 게 확인됐다.

그래서 이름을 포기하고, UNI2-h(공식 스펙) frozen feature를 라벨 없이 k-means로 군집화한 뒤
(data/fit_clusters_uni2native.py), 환자별 "군집 비율"(K차원 히스토그램, data/
compute_cluster_histograms_uni2native.py로 사전 계산) 자체를 clinical/RNA와 똑같은 방식으로
Cox 가산항에 직접 넣는다 — "이 군집이 뭔지" 우리가 판정하지 않고, 어느 군집 비율이 hazard와
상관있는지는 152개 생존 라벨 자체가 학습 중에 결정하게 둔다.

M7과의 유일한 차이: risk = risk_head(z_rna) + clinical_linear(clin_raw) + hist_linear(cluster_hist)
— hist_linear는 clinical_linear와 동일한 관례(bias 없음, zero-init, raw feature 직결) —
파라미터가 K개뿐이라 이 코호트 크기(152명)에서 과적합 위험이 attention/MLP 대비 훨씬 작다.
"""
import torch
import torch.nn as nn

from .clinical_encoder import STAGE_FIELDS, _STAGE_BUFFER_NAMES
from .rna_encoder import RNAEncoder
from config import ModelConfig

RNA_EMBED_DIM = 64


class HDP(nn.Module):
    def __init__(self, cfg: ModelConfig, age_mean: float, age_std: float, rna_input_dim: int,
                 hist_dim: int, rna_dim: int | None = None,
                 use_margin: bool = False, margin_stats: tuple[float, float] | None = None,
                 use_age_sex: bool = True,
                 use_staging: bool = False, stage_stats: dict[str, tuple[float, float]] | None = None):
        super().__init__()
        if use_margin and margin_stats is None:
            raise ValueError("use_margin=True면 margin_stats가 필요합니다.")
        if use_staging and stage_stats is None:
            raise ValueError("use_staging=True면 stage_stats가 필요합니다.")
        self.use_margin = use_margin
        self.use_age_sex = use_age_sex
        self.use_staging = use_staging
        self.hist_dim = hist_dim

        rna_dim = rna_dim or RNA_EMBED_DIM
        self.rna_encoder = RNAEncoder(rna_input_dim, rna_dim, dropout=0.3)
        self.risk_head = nn.Sequential(nn.LayerNorm(rna_dim), nn.Linear(rna_dim, 1))

        # clinical cox_add — models/clinical_rna_only.py::ClinicalRNAOnly(combine_mode="cox_add")와
        # 동일 관례(raw z-score feature 직결, bias 없는 zero-init linear).
        self.register_buffer("age_mean", torch.tensor(age_mean, dtype=torch.float32))
        self.register_buffer("age_std", torch.tensor(age_std, dtype=torch.float32))
        if use_margin:
            m_mean, m_std = margin_stats
            self.register_buffer("margin_mean", torch.tensor(m_mean, dtype=torch.float32))
            self.register_buffer("margin_std", torch.tensor(m_std, dtype=torch.float32))
        if use_staging:
            for field in STAGE_FIELDS:
                mean, std = stage_stats[field]
                short = _STAGE_BUFFER_NAMES[field]
                self.register_buffer(f"{short}_mean", torch.tensor(mean, dtype=torch.float32))
                self.register_buffer(f"{short}_std", torch.tensor(std, dtype=torch.float32))
        raw_dim = (2 if use_age_sex else 0) + (2 if use_margin else 0) + (2 * len(STAGE_FIELDS) if use_staging else 0)
        self.clinical_linear = nn.Linear(raw_dim, 1, bias=False)
        nn.init.zeros_(self.clinical_linear.weight)

        # WSI 유래 군집 히스토그램(K차원, 비율 합=1) -> Cox 가산항. clinical_linear와 동일 관례.
        self.hist_linear = nn.Linear(hist_dim, 1, bias=False)
        nn.init.zeros_(self.hist_linear.weight)

    def _clinical_raw(self, age_years: torch.Tensor, sex_idx: torch.Tensor,
                       margin_ord: torch.Tensor | None = None,
                       stage_ord: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """models/clinical_rna_only.py::ClinicalRNAOnly._clinical_embed(cox_add 분기)와 동일."""
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

    def forward(self, age_years: torch.Tensor, sex_idx: torch.Tensor, rna: torch.Tensor,
                cluster_hist: torch.Tensor,
                margin_ord: torch.Tensor | None = None,
                stage_ord: dict[str, torch.Tensor] | None = None,
                return_components: bool = False):
        """
        Args:
            age_years:    () 스칼라
            sex_idx:      () 스칼라
            rna:          (G,) 코호트 내부 z-score
            cluster_hist: (K,) 환자의 군집 비율(합=1) — data/compute_cluster_histograms_uni2native.py 산출
            margin_ord/stage_ord: ClinicalRNAOnly와 동일 규약
            return_components: True면 risk 대신 {"rna":.., "clin":.., "hist":..} 항별 dict 반환
                                (2026-09-01, scripts/diagnose_hdp_checkpoint_weights.py용 — branch별
                                실제 기여도를 뜯어보기 위해 추가. 기본 False로 기존 동작 그대로.)
        Returns:
            risk: (1,) (return_components=False) 또는 항별 dict(각 (1,))
        """
        z_r = self.rna_encoder(rna.unsqueeze(0)).squeeze(0)
        risk_rna = self.risk_head(z_r.unsqueeze(0)).view(1)

        clin_raw = self._clinical_raw(age_years, sex_idx, margin_ord, stage_ord=stage_ord)
        risk_clin = self.clinical_linear(clin_raw).view(1)

        risk_hist = self.hist_linear(cluster_hist.float().unsqueeze(0)).view(1)

        if return_components:
            return {"rna": risk_rna, "clin": risk_clin, "hist": risk_hist}
        return risk_rna + risk_clin + risk_hist
