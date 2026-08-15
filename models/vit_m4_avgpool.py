"""
ViT_M4_AvgPool — ViT_M4(vit_m4.py)에서 RNA-guided ABMIL(AttentionPooling)을 학습 파라미터가
전혀 없는 단순 평균 풀링으로 교체한 ablation. train.py의 --M4 --avgpool로 선택된다.

2026-08-14: PMA(4-component+co-attention) -> M4(단일 gated-ABMIL) -> M4+skip-patch-vit로
WSI pooling/patch-mixing 구조를 계속 단순화해도 external이 전혀 안 바뀌는 것과 별개로,
diagnose_pma_wsi_structure.py가 patch attention entropy가 ~0.999(사실상 uniform)이고 attn_pool
자신의 gradient norm이 다른 모듈 대비 100~250배 작다는 걸 실측으로 확인했다 — attention이
"무언가를 배우고 있다"는 전제 자체가 깨져 있다는 뜻이다. 이미 사실상 균일하게 동작하는
학습된 attention 대신, 애초에 학습 파라미터가 없는 평균 풀링으로 바꿔서 (a) 성능이 정말
동일한지(예상대로면 그래야 함 — 이미 균일했으니), (b) 이 쓸모없는 게이트 파라미터가 유발하는
불필요한 gradient 노이즈가 사라지면서 seed 간 변동성(std)이 줄어드는지를 함께 본다.

vit_m1_avgpool.py::ViT_M1_AvgPool과 달리 attn_dispersion/skip_patch_vit까지 포함한 현재
표준 레시피와 완전히 호환되도록 forward()를 새로 구성한다(ViT_M1_AvgPool은 이 레시피
이전에 만들어져 이 옵션들을 지원하지 않는다).
"""
import torch

from .spatial_features import attention_dispersion
from .vit_m4 import ViT_M4
from config import ModelConfig


class ViT_M4_AvgPool(ViT_M4):
    def __init__(self, cfg: ModelConfig, age_mean: float, age_std: float, rna_input_dim: int,
                 precomputed: bool = True, backbone: str = "resnet50", use_staging: bool = False,
                 stage_stats: dict[str, tuple[float, float]] | None = None, use_margin: bool = False,
                 margin_stats: tuple[float, float] | None = None, use_age_sex: bool = True,
                 combine_mode: str = "concat", use_attn_dispersion: bool = False,
                 skip_patch_vit: bool = False):
        super().__init__(cfg, age_mean, age_std, rna_input_dim, precomputed, backbone,
                          use_staging, stage_stats, use_margin, margin_stats, use_age_sex,
                          combine_mode, use_attn_dispersion, skip_patch_vit)
        del self.attn_pool  # RNA-guided 게이트 파라미터 제거 — 평균 풀링은 학습 파라미터가 없음

    def forward(
        self,
        coords: torch.Tensor,
        patch_paths=None,
        features: torch.Tensor | None = None,
        transform=None,
        chunk_size: int | None = None,
        rna_context: torch.Tensor | None = None,  # 무시 — 평균 풀링은 RNA로 조건화되지 않음
        tile_cache: dict | None = None,
        tumor_type: torch.Tensor | None = None,
    ) -> dict:
        patch_tokens = self._patch_tokens(coords, patch_paths, features, transform, chunk_size, tile_cache)
        ctx_tokens   = patch_tokens if self.skip_patch_vit else self.vit(patch_tokens, coords, tumor_type=tumor_type)
        wsi_embed    = ctx_tokens.mean(dim=0)  # (D,) — 무가중 평균, RNA/attention 개입 없음
        attn_weights = torch.full(
            (ctx_tokens.shape[0],), 1.0 / ctx_tokens.shape[0], device=ctx_tokens.device
        )
        out = {"embed": wsi_embed, "attn_weights": attn_weights, "meanpool_embed": wsi_embed}
        if self.use_attn_dispersion:
            # attn_weights가 이미 균일하므로 attention_dispersion()은 사실상 "패치들이 슬라이드
            # 안에 얼마나 넓게 퍼져 있는지"(비가중 공간 표준편차)만 남는다 — attn_pool 제거와
            # 별개로 spatial_feat 자체는 여전히 계산 가능하다.
            out["spatial_feat"] = attention_dispersion(coords, attn_weights) * self.dispersion_scale
        return out
