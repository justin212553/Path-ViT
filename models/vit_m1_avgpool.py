"""
ViT_M1_AvgPool — ViT_M1(vit_m1.py)에서 ABMIL(AttentionPooling)을 단순 평균 풀링으로
교체한 ablation 변형. train.py의 --avgpool 플래그로 선택된다.

ABMIL의 게이트 네트워크(attn_v, attn_u, attn_w)가 가진 학습 파라미터를 완전히 제거해,
"패치 → WSI 집계" 단계의 표현력/용량을 낮춘다 — "어떤 패치가 중요한가"를 학습하는 능력을
포기하는 대신, 학습 파라미터가 하나도 없는 산술 평균으로 N개 패치 토큰을 WSI 임베딩
1개로 합친다. capacity 축소가 seed 간 불안정성을 줄이는지 확인하기 위한 대조군이다.
"""
import torch

from .vit_m1 import ViT_M1
from config import ModelConfig


class ViT_M1_AvgPool(ViT_M1):
    def __init__(self, cfg: ModelConfig, precomputed: bool = True):
        super().__init__(cfg, precomputed)
        del self.attn_pool  # 게이트 파라미터 제거 — 평균 풀링은 학습 파라미터가 없음

    def forward(
        self,
        coords: torch.Tensor,
        patch_paths=None,
        features: torch.Tensor | None = None,
        transform=None,
        chunk_size: int | None = None,
    ) -> dict:
        patch_tokens = self._patch_tokens(coords, patch_paths, features, transform, chunk_size)
        ctx_tokens   = self.vit(patch_tokens, coords)                       # (N, D)
        wsi_embed    = ctx_tokens.mean(dim=0)                               # (D,) — 무가중 평균
        attn_weights = torch.full(
            (ctx_tokens.shape[0],), 1.0 / ctx_tokens.shape[0], device=ctx_tokens.device
        )
        return {"embed": wsi_embed, "attn_weights": attn_weights}
