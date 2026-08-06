"""
ViT_M2_Pool — M2(ABMIL+Late Fusion concat)와 PMA(다성분 pooling+co-attention) 사이의 pooling
방식 불일치를 통제하기 위한 ablation. train.py --M2_POOL.

ViT_M1_Pool(학습되는 고정 query)의 자연스러운 다음 단계 — 여기서는 z_clinical(ClinicalEncoder
출력)을 co-attention query로 써서, PMA가 z_rna로 하는 것과 대칭적으로 "clinical이 WSI의 4개
pooling 관점 중 어디를 봐야 할지 실제로 guide할 수 있는가"를 검증한다. margin(residual_disease,
--clinical-margin)이 양쪽 코호트에서 독립적으로 유의한 진짜 신호로 확인된 뒤 처음 시도하는
조합 — 이전에는 clinical(age/sex뿐)이 noise 수준이라 이 실험이 의미가 없었다.
"""
import torch
import torch.nn as nn

from .vit_m1 import ViT_M1
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
    ):
        super().__init__(cfg, precomputed, backbone, use_attn_dispersion=use_attn_dispersion)
        self.use_margin = use_margin
        self.use_age_sex = use_age_sex
        self.multi_pool = MultiComponentPooling(cfg.embed_dim)
        self.clinical_encoder = ClinicalEncoder(
            cfg.embed_dim, age_mean, age_std, use_margin=use_margin, margin_stats=margin_stats,
            use_age_sex=use_age_sex,
        )
        self.component_coattn = CoAttentionPooling(
            cfg.embed_dim, num_heads=num_heads, dropout=cfg.dropout, context_dim=cfg.embed_dim
        )
        del self.attn_pool  # ViT_M1의 단일-벡터 ABMIL은 안 쓴다(multi_pool로 대체)

        # risk_head 입력: [z_wsi(co-attention 결과), z_clinical] — PMA가 [z_wsi, z_clinical, z_rna]
        # 를 쓰는 것과 같은 관례(query로 쓴 모달리티를 risk_head에도 concat).
        risk_input_dim = cfg.embed_dim * 2 + (1 if use_attn_dispersion else 0)
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
    ) -> dict:
        patch_tokens = self._patch_tokens(coords, patch_paths, features, transform, chunk_size, tile_cache)
        ctx_tokens = self.vit(patch_tokens, coords)
        components, attn_weights = self.multi_pool(ctx_tokens)  # (4, D), (N,)
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
        clinical_kwargs = {} if margin_ord is None else {"margin_ord": margin_ord.unsqueeze(0)}
        z_clinical = self.clinical_encoder(
            age_years.unsqueeze(0), sex_idx.unsqueeze(0), **clinical_kwargs
        ).squeeze(0)  # (D,)
        z_wsi, _ = self.component_coattn(patient_embed, z_clinical)  # (D,) — clinical이 4개 관점 중 골라 가중합
        fused = torch.cat([z_wsi, z_clinical], dim=-1)  # (2D,)
        if spatial_feat is not None:
            fused = torch.cat([fused, spatial_feat], dim=-1)
        return fused
