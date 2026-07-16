"""
ViT_M4B — pre-ViT FiLM token conditioning ablation, train.py --M4B

ViT_M4(FiLM bias-in-gate)/ViT_M4A(co-attention pooling)와 fusion 골격은 동일하다
(encode_rna() → combine_with_clinical_rna()에서 [z_wsi ‖ z_clinical ‖ z_rna] concat →
risk_head). RNA가 개입하는 "지점"만 다르다:
  - ViT_M4  : ViT를 지난 *뒤* ABMIL 게이트(tanh·sigmoid) pre-activation에 z_rna를 더한다.
  - ViT_M4A : ViT를 지난 *뒤* 풀링 자체를 z_rna-query co-attention으로 대체한다.
  - ViT_M4B : ViT를 지나기 *전*, CNN 직후의 패치 토큰 자체를 z_rna로 FiLM(scale+shift)
    조건화한다 — self-attention이 이미 RNA-informed된 토큰 위에서 계산되므로, "어떤
    패치가 어떤 패치를 주목하는가" 자체가 RNA 영향을 받는다. 지금까지 시도한 세 가지
    중 가장 이른 지점에서 개입하는 버전.

RNA는 이미 FiLM으로 토큰에 주입됐으므로, attn_pool은 context 없는 일반 ABMIL로 되돌린다
(M4/M4A와 마찬가지로 "개입 지점 하나만 다르다"는 통제된 비교를 유지하기 위함 — 이중으로
개입시키면 어느 지점의 효과인지 구분할 수 없다).
"""
import torch
import torch.nn as nn

from .vit_m1 import AttentionPooling
from .vit_m4 import ViT_M4
from config import ModelConfig


class FiLMTokenConditioning(nn.Module):
    """RNA 컨텍스트로 ViT 입력 패치 토큰에 FiLM(scale+shift)을 적용.

    [gamma/beta 초기화] 무작위 초기화로 시작하면 학습 초반 토큰이 심하게 뒤틀려
    최적화가 불안정해질 수 있다(환자 90~150명 규모에서 특히 리스크가 크다). gamma_proj는
    weight=0, bias=1로, beta_proj는 weight=0, bias=0으로 초기화해 학습 시작 시 FiLM이
    항등변환(identity)에서 출발하게 한다(AdaLN-Zero, DiT 계열과 동일 관례) — 이후
    gradient가 필요한 만큼만 조건화를 학습한다.
    """

    def __init__(self, embed_dim: int, context_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(context_dim, embed_dim)
        self.beta_proj = nn.Linear(context_dim, embed_dim)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens:  (N, D) — CNN을 지난 패치 토큰 (ViT 통과 전)
            context: (D,)   — RNA 임베딩(z_rna). 슬라이드 내 모든 패치에 동일하게 broadcast
                     (환자 단위 정보라 패치마다 다르지 않음)
        Returns:
            (N, D) — FiLM 조건화된 패치 토큰
        """
        gamma = self.gamma_proj(context)  # (D,)
        beta = self.beta_proj(context)    # (D,)
        return tokens * gamma + beta


class ViT_M4B(ViT_M4):
    """
    ViT_M4/ViT_M4A와 동일한 3-모달 Late Fusion 골격에서, RNA 개입 지점을 ViT *이전*
    (patch token 자체)으로 옮긴 ablation.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        age_mean: float,
        age_std: float,
        rna_input_dim: int,
        precomputed: bool = True,
        backbone: str = "resnet50",
    ):
        super().__init__(cfg, age_mean, age_std, rna_input_dim, precomputed, backbone)
        self.film = FiLMTokenConditioning(cfg.embed_dim, cfg.embed_dim)
        self.attn_pool = AttentionPooling(cfg.embed_dim)  # context 없는 일반 ABMIL로 복귀

    def forward(
        self,
        coords: torch.Tensor,
        patch_paths=None,
        features: torch.Tensor | None = None,
        transform=None,
        chunk_size: int | None = None,
        rna_context: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            rna_context: (D,) — FiLM 조건화용 컨텍스트. ViT_M4/ViT_M4A와 달리 attn_pool이
                         아니라 ViT 입력 토큰에 적용된다.
        Returns:
            embed:        (D,) — WSI 임베딩
            attn_weights: (N_patches,)
        """
        patch_tokens = self._patch_tokens(coords, patch_paths, features, transform, chunk_size)
        if rna_context is not None:
            patch_tokens = self.film(patch_tokens, rna_context)
        ctx_tokens = self.vit(patch_tokens, coords)
        wsi_embed, attn_weights = self.attn_pool(ctx_tokens, context=None)
        return {"embed": wsi_embed, "attn_weights": attn_weights}
