"""
ViT_M2_Pool — M2(ABMIL+Late Fusion concat)와 PMA(다성분 pooling+co-attention) 사이의 pooling
방식 불일치를 통제하기 위한 ablation. train.py --M2_POOL.

ViT_M1_Pool(학습되는 고정 query)의 자연스러운 다음 단계 — 여기서는 z_clinical(ClinicalEncoder
출력)을 co-attention query로 써서, PMA가 z_rna로 하는 것과 대칭적으로 "clinical이 WSI의 4개
pooling 관점 중 어디를 봐야 할지 실제로 guide할 수 있는가"를 검증한다. margin(residual_disease,
--clinical-margin)이 양쪽 코호트에서 독립적으로 유의한 진짜 신호로 확인된 뒤 처음 시도하는
조합 — 이전에는 clinical(age/sex뿐)이 noise 수준이라 이 실험이 의미가 없었다.

2026-08-07: pooling_mode/combine_mode 두 축을 추가했다. UNI+age/sex만으로 co-attention query를
돌려보니 external이 0.49~0.51(사실상 랜덤)에 몰려, M1_POOL(clinical 없이 self-attention만 쓴
버전)의 0.556보다도 못한 결과가 나왔다 — age/sex는 RNA와 달리 신호가 약해 "어느 관점을 볼지"를
잘못된 기준으로 고르게 만드는 것으로 추정된다. 그래서 pooling 자체는 M1_POOL과 동일한
self-attention(clinical 개입 없음)으로 하고, clinical은 ViT_PMA/ViT_M2와 동일한 관례로 pooling과
분리해 risk_head 스칼라에 고전적 Cox 가산항(cox_add)으로만 더하는 조합을 추가로 지원한다.
  - pooling_mode="coattn"(기본, 기존 동작): z_clinical이 4개 관점의 co-attention query.
  - pooling_mode="selfattn": 4개 관점이 서로 self-attention(models/vit_m1_pool.py와 동일 모듈
    재사용) — clinical은 pooling에 전혀 관여하지 않는다.
  - combine_mode="concat"(기본, 기존 동작): z_clinical을 risk_head 입력에 concat.
  - combine_mode="cox_add": clinical을 임베딩하지 않고 risk_head(z_wsi) 스칼라에 zero-init
    가산항으로 직접 더한다(train.py의 공용 cox_add 처리 블록, ViT_PMA/ViT_M2와 동일 관례).
"""
import torch
import torch.nn as nn

from .vit_m1 import ViT_M1
from .vit_m1_pool import SelfAttentionPooling
from .vit_m4a import CoAttentionPooling
from .multi_component_pooling import MultiComponentPooling
from .clinical_encoder import ClinicalEncoder
from .spatial_features import attention_dispersion
from config import ModelConfig


class ViT_M2_Pool(ViT_M1):
    def __init__(
        self,
        cfg: ModelConfig,
        age_mean: float,
        age_std: float,
        precomputed: bool = True,
        backbone: str = "resnet50",
        num_heads: int = 2,
        use_margin: bool = False,
        margin_stats: tuple[float, float] | None = None,
        use_age_sex: bool = True,
        use_attn_dispersion: bool = False,
        pooling_mode: str = "coattn",
        combine_mode: str = "concat",
    ):
        super().__init__(cfg, precomputed, backbone, use_attn_dispersion=use_attn_dispersion)
        if pooling_mode not in ("coattn", "selfattn"):
            raise ValueError(f"알 수 없는 pooling_mode: {pooling_mode}")
        if combine_mode not in ("concat", "cox_add"):
            raise ValueError(f"알 수 없는 combine_mode: {combine_mode}")
        self.pooling_mode = pooling_mode
        self.combine_mode = combine_mode
        self.use_margin = use_margin
        self.use_age_sex = use_age_sex
        self.multi_pool = MultiComponentPooling(cfg.embed_dim)
        del self.attn_pool  # ViT_M1의 단일-벡터 ABMIL은 안 쓴다(multi_pool로 대체)

        # z_clinical 임베딩은 (a) co-attention query로 쓰거나 (b) concat할 때만 필요.
        # cox_add+selfattn 조합이면 아예 안 만든다(raw feature를 _clinical_raw()로 직접 씀).
        need_clinical_encoder = (pooling_mode == "coattn") or (combine_mode == "concat")
        if need_clinical_encoder:
            self.clinical_encoder = ClinicalEncoder(
                cfg.embed_dim, age_mean, age_std, use_margin=use_margin, margin_stats=margin_stats,
                use_age_sex=use_age_sex,
            )

        if pooling_mode == "coattn":
            self.component_coattn = CoAttentionPooling(
                cfg.embed_dim, num_heads=num_heads, dropout=cfg.dropout, context_dim=cfg.embed_dim
            )
        else:
            self.component_selfattn = SelfAttentionPooling(cfg.embed_dim, num_heads=num_heads, dropout=cfg.dropout)

        if combine_mode == "cox_add":
            # models/vit_m2.py::ViT_M2 cox_add 분기와 동일 관례.
            self.register_buffer("age_mean", torch.tensor(age_mean, dtype=torch.float32))
            self.register_buffer("age_std", torch.tensor(age_std, dtype=torch.float32))
            if use_margin:
                m_mean, m_std = margin_stats
                self.register_buffer("margin_mean", torch.tensor(m_mean, dtype=torch.float32))
                self.register_buffer("margin_std", torch.tensor(m_std, dtype=torch.float32))
            raw_dim = (2 if use_age_sex else 0) + (2 if use_margin else 0)
            if raw_dim == 0:
                raise ValueError("use_age_sex=False이고 use_margin=False면 clinical 입력이 없습니다.")
            self.clinical_linear = nn.Linear(raw_dim, 1, bias=False)
            nn.init.zeros_(self.clinical_linear.weight)  # 초기엔 clinical 없는 M1_POOL과 동일

        # risk_head 입력: [z_wsi(D)] (+ z_clinical(D), combine_mode="concat"일 때만) (+ dispersion(1,))
        risk_input_dim = cfg.embed_dim + (cfg.embed_dim if combine_mode == "concat" else 0) + \
            (1 if use_attn_dispersion else 0)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(risk_input_dim),
            nn.Linear(risk_input_dim, 1),
        )

    def forward(
        self,
        coords: torch.Tensor,
        patch_paths=None,
        features: torch.Tensor | None = None,
        transform=None,
        chunk_size: int | None = None,
        rna_context: torch.Tensor | None = None,  # 사용 안 함(train.py 호출 시그니처 호환용)
        tile_cache: dict | None = None,
        tumor_type: torch.Tensor | None = None,  # 사용 안 함(train.py 호출 시그니처 호환용)
    ) -> dict:
        patch_tokens = self._patch_tokens(coords, patch_paths, features, transform, chunk_size, tile_cache)
        ctx_tokens = self.vit(patch_tokens, coords)
        components, attn_weights, _ = self.multi_pool(ctx_tokens)  # (4, D), (N,), risk_stats(미사용)
        out = {
            "embed": components, "attn_weights": attn_weights,
            "meanpool_embed": ctx_tokens.mean(dim=0),
        }
        if self.use_attn_dispersion:
            out["spatial_feat"] = attention_dispersion(coords, attn_weights) * self.dispersion_scale
        return out

    def combine_with_clinical_pool(
        self,
        patient_embed: torch.Tensor,  # (4, D) — 환자 단위로 평균 풀링된 4개 관점
        age_years: torch.Tensor,
        sex_idx: torch.Tensor,
        margin_ord: torch.Tensor | None = None,  # self.use_margin=True일 때만 필요
        spatial_feat: torch.Tensor | None = None,  # (spatial_feat_dim,)
    ) -> torch.Tensor:
        z_clinical = None
        if hasattr(self, "clinical_encoder"):
            clinical_kwargs = {} if margin_ord is None else {"margin_ord": margin_ord.unsqueeze(0)}
            z_clinical = self.clinical_encoder(
                age_years.unsqueeze(0), sex_idx.unsqueeze(0), **clinical_kwargs
            ).squeeze(0)  # (D,)

        if self.pooling_mode == "coattn":
            z_wsi, _ = self.component_coattn(patient_embed, z_clinical)  # clinical이 4개 관점 중 골라 가중합
        else:
            z_wsi = self.component_selfattn(patient_embed)  # clinical 개입 없음(models/vit_m1_pool.py와 동일)

        if self.combine_mode == "cox_add":
            fused = z_wsi  # clinical은 여기서 안 섞이고, 호출부(train.py)가 _clinical_raw()로 별도 계산해 더함
        else:
            fused = torch.cat([z_wsi, z_clinical], dim=-1)  # (2D,)
        if spatial_feat is not None:
            fused = torch.cat([fused, spatial_feat], dim=-1)
        return fused

    def _clinical_raw(self, age_years: torch.Tensor, sex_idx: torch.Tensor,
                       margin_ord: torch.Tensor | None = None,
                       stage_ord: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """combine_mode="cox_add" 전용 — models/vit_m2.py::ViT_M2._clinical_raw와 동일 관례.
        stage_ord는 M2_POOL이 staging을 지원하지 않아 시그니처 호환용으로만 받는다(train.py의
        공용 dispatch가 모든 cox_add 모델에 동일한 인자로 호출한다)."""
        feats = []
        if self.use_age_sex:
            age_z = (age_years.float() - self.age_mean) / self.age_std
            feats += [age_z, sex_idx.float()]
        if self.use_margin:
            ordv = margin_ord.float()
            known = (ordv >= 0).float()
            z = torch.where(ordv >= 0, (ordv - self.margin_mean) / self.margin_std, torch.zeros_like(ordv))
            feats += [z, known]
        return torch.stack(feats, dim=-1).unsqueeze(0)  # (1, raw_dim)
