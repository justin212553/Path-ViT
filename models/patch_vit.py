"""
PatchViT — CAMELYON17 림프절 미세전이 탐지 모델 (순수 MIL)

패치 → CNN → 공간 임베딩 ViT → CLS 토큰 → 슬라이드 레벨 분류

Forward 출력:
    slide_logits : (1, 2) — 슬라이드 전이 여부 logit (N0 / N1+)
"""
import torch
import torch.nn as nn

from .cnn_encoder import CNNEncoder
from .vit_encoder import ViTEncoder
from config import ModelConfig


class PatchViT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cnn = CNNEncoder(cfg.cnn_backbone, cfg.embed_dim)
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
            slide_logits: (1, 2)
        """
        patch_tokens = self.cnn(patches)               # (N, D)
        h_img, _     = self.vit(patch_tokens, coords)  # (1, D)
        return {"slide_logits": self.classifier(h_img)}
