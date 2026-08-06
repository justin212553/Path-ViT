"""
ViT_M1_Pool — M1(ABMIL 단일 벡터)과 M3/PMA(다성분 pooling+co-attention) 사이의 pooling
방식 불일치를 통제하기 위한 ablation. train.py --M1_POOL.

M3(WSI+RNA)/PMA(WSI+RNA+Clinical)와 M1(WSI 단독)을 비교할 때, 지금까지는 "RNA/Clinical
유무"와 "WSI pooling 방식(ABMIL vs MultiComponentPooling+co-attention)"이 동시에 바뀌어
있었다 — 두 요인이 뒤섞인 비교였다. 이 모델은 M1과 같은 WSI 단독 입력에 M3/PMA와 동일한
MultiComponentPooling(mean/std/attn-weighted/top-k)을 적용하되, RNA/clinical처럼 query로
쓸 외부 모달리티가 없으므로 학습되는 고정 파라미터를 query로 쓴다(ViT의 [CLS] 토큰, DETR의
object query와 같은 개념 — 모든 환자에 대해 동일한 벡터지만, key/value(그 환자의 4개 관점)는
환자마다 다르므로 attention 가중치와 최종 출력은 여전히 환자별로 다르다). 이렇게 하면
"co-attention 구조 자체의 효과"와 "RNA/clinical이 그 co-attention을 guide하는 효과"를 분리해서
볼 수 있다.
"""
import torch
import torch.nn as nn

from .vit_m1 import ViT_M1
from .vit_m4a import CoAttentionPooling
from .multi_component_pooling import MultiComponentPooling
from .spatial_features import attention_dispersion
from config import ModelConfig


class ViT_M1_Pool(ViT_M1):
    def __init__(
        self,
        cfg: ModelConfig,
        precomputed: bool = True,
        backbone: str = "resnet50",
        num_heads: int = 2,
        use_attn_dispersion: bool = False,
    ):
        super().__init__(cfg, precomputed, backbone, use_attn_dispersion=use_attn_dispersion)
        self.multi_pool = MultiComponentPooling(cfg.embed_dim)
        self.component_coattn = CoAttentionPooling(
            cfg.embed_dim, num_heads=num_heads, dropout=cfg.dropout, context_dim=cfg.embed_dim
        )
        # 모든 환자 공통, 학습되는 co-attention query — RNA/clinical 없이 "어떤 관점을 봐야
        # 하는지"를 데이터가 아니라 학습으로 고정시킨다(PMA의 z_rna 자리를 대신함).
        self.learned_query = nn.Parameter(torch.randn(cfg.embed_dim) * 0.02)
        del self.attn_pool  # ViT_M1의 단일-벡터 ABMIL은 안 쓴다(multi_pool로 대체)

        risk_input_dim = cfg.embed_dim + (1 if use_attn_dispersion else 0)
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

    def pool_components(self, patient_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patient_embed: (4, D) — 환자 단위로 평균 풀링된 4개 관점(train.py에서 계산)
        Returns:
            z_wsi: (D,) — 학습된 고정 query로 co-attention한 결과
        """
        z_wsi, _ = self.component_coattn(patient_embed, self.learned_query)
        return z_wsi
