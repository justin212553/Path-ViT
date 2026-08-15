"""
ViT_M1_Pool — M1(ABMIL 단일 벡터)과 M3/PMA(다성분 pooling+co-attention) 사이의 pooling
방식 불일치를 통제하기 위한 ablation. train.py --M1_POOL.

M3(WSI+RNA)/PMA(WSI+RNA+Clinical)와 M1(WSI 단독)을 비교할 때, 지금까지는 "RNA/Clinical
유무"와 "WSI pooling 방식(ABMIL vs MultiComponentPooling+co-attention)"이 동시에 바뀌어
있었다 — 두 요인이 뒤섞인 비교였다. 이 모델은 M1과 같은 WSI 단독 입력에 M3/PMA와 동일한
MultiComponentPooling(mean/std/attn-weighted/top-k)을 적용한다.

2026-08-07: 초판(학습되는 고정 query로 co-attention)에서 self-attention으로 교체했다 — M1엔
RNA/clinical처럼 "어디를 봐야 하는지" 알려줄 외부 모달리티가 없으니, 고정 query를 인공적으로
학습시키는 것보다 4개 관점이 서로를 직접 attend하게 하는 편이 구조적으로 더 자연스럽다(M2_POOL
이 clinical을 진짜 외부 query로 쓰는 것과 대비되는 지점 — M1은 외부 query가 없다는 사실 자체를
구조에 반영).
"""
import torch
import torch.nn as nn

from .vit_m1 import ViT_M1
from .multi_component_pooling import MultiComponentPooling
from .spatial_features import attention_dispersion
from config import ModelConfig


class SelfAttentionPooling(nn.Module):
    """
    4개 pooling 관점(components)이 서로를 attend하는 self-attention — 외부 query 없이
    관점들끼리 정보를 교환한 뒤 평균해 하나의 벡터로 합친다. 표준 Transformer encoder layer
    1개(MHA + residual/LN + FFN + residual/LN)와 동일한 구조.
    """

    def __init__(self, embed_dim: int, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, components: torch.Tensor) -> torch.Tensor:
        """
        Args:
            components: (4, D) — 환자 단위로 평균 풀링된 4개 관점
        Returns:
            z_wsi: (D,) — self-attention을 거친 4개 관점의 평균
        """
        x = components.unsqueeze(0)  # (1, 4, D)
        attn_out, _ = self.mha(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(0).mean(dim=0)  # (D,)


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
        self.component_selfattn = SelfAttentionPooling(cfg.embed_dim, num_heads=num_heads, dropout=cfg.dropout)
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
        components, attn_weights, _ = self.multi_pool(ctx_tokens)  # (4, D), (N,), risk_stats(미사용)
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
            z_wsi: (D,) — self-attention으로 관점들을 섞은 결과
        """
        return self.component_selfattn(patient_embed)
