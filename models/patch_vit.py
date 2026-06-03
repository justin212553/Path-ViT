"""
PatchViT — CAMELYON17 림프절 미세전이 탐지 모델

패치 → CNN → 공간 임베딩 ViT → 패치별 분류

Forward 출력:
    patch_logits : (N_patches, 2) — 패치별 종양 여부 logit (정상 / 종양)
"""
import torch
import torch.nn as nn

from .cnn_encoder import CNNEncoder
from .vit_encoder import ViTEncoder
from config import ModelConfig


class PatchViT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cnn = CNNEncoder(cfg.embed_dim)
        self.vit = ViTEncoder(cfg.embed_dim, cfg.num_heads,
                              cfg.num_transformer_layers, cfg.dropout,
                              cfg.max_grid_size)

        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, 2),
        )

    def forward(self, patches: torch.Tensor, coords: torch.Tensor) -> dict:
        """
        Args:
            patches: (N_patches, 3, H, W)
            coords:  (N_patches, 2)
        Returns:
            patch_logits: (N_patches, 2)
        """
        patch_tokens = self.cnn(patches)              # (N, D)
        ctx_tokens   = self.vit(patch_tokens, coords) # (N, D)
        return {"patch_logits": self.classifier(ctx_tokens)}
