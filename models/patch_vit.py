"""
PatchViT — CAMELYON17 림프절 미세전이 탐지 모델 (WSI 단위 MIL)

패치 → CNN → 공간 임베딩 ViT(self-attention) → attention pooling → WSI 단위 분류

Forward 출력:
    wsi_logits   : (1, 2)        — WSI(노드) 단위 전이 여부 logit (정상 / 전이)
    attn_weights : (N_patches,)  — 패치별 attention 가중치 (시각화용)
"""
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from .cnn_encoder import CNNEncoder
from .vit_encoder import ViTEncoder
from config import ModelConfig


class AttentionPooling(nn.Module):
    """
    Gated attention pooling (Ilse et al., 2018 ABMIL).
    패치 토큰 집합 (N, D) → 가중합으로 단일 WSI 임베딩 (D,) 집계.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attn_v = nn.Linear(embed_dim, hidden_dim)
        self.attn_u = nn.Linear(embed_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (N, D) — 패치 토큰
        Returns:
            wsi_embed:    (D,) — attention 가중합으로 집계된 WSI 임베딩
            attn_weights: (N,) — 패치별 attention 가중치 (합=1)
        """
        gate = torch.tanh(self.attn_v(tokens)) * torch.sigmoid(self.attn_u(tokens))  # (N, H)
        scores = self.attn_w(gate).squeeze(-1)        # (N,)
        attn_weights = torch.softmax(scores, dim=0)   # (N,)
        wsi_embed = (attn_weights.unsqueeze(-1) * tokens).sum(dim=0)  # (D,)
        return wsi_embed, attn_weights


class PatchViT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cnn = CNNEncoder(cfg.embed_dim)
        self.vit = ViTEncoder(cfg.embed_dim, cfg.num_heads,
                              cfg.num_transformer_layers, cfg.dropout,
                              cfg.max_grid_size, num_landmarks=cfg.num_landmarks)
        self.attn_pool = AttentionPooling(cfg.embed_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, 2),
        )

    def forward(
        self,
        patch_paths: list[Path],
        coords: torch.Tensor,
        transform,
        chunk_size: int | None = None,
    ) -> dict:
        """
        Args:
            patch_paths: N개 패치 이미지 파일 경로 — 이미지 디코딩을 chunk_size 단위로
                         지연 로딩해 한 번에 메모리에 올리는 패치 수를 제한한다
                         (패치 수에 cap이 없는 대형 WSI에서 host RAM OOM 방지)
            coords:      (N_patches, 2)
            transform:   패치 이미지 → 텐서 변환 (CAMELYON17NodeDataset.transform)
            chunk_size:  CNN을 이 크기 단위로 나눠 실행. None이면 한 번에 실행
        Returns:
            wsi_logits:   (1, 2)
            attn_weights: (N_patches,)
        """
        device = coords.device
        chunk_size = chunk_size or len(patch_paths)

        patch_tokens = torch.cat([
            self.cnn(
                torch.stack([
                    transform(Image.open(p).convert("RGB"))
                    for p in patch_paths[i : i + chunk_size]
                ]).to(device, non_blocking=True)
            )
            for i in range(0, len(patch_paths), chunk_size)
        ])
        ctx_tokens   = self.vit(patch_tokens, coords)          # (N, D)
        wsi_embed, attn_weights = self.attn_pool(ctx_tokens)   # (D,), (N,)
        wsi_logits = self.classifier(wsi_embed.unsqueeze(0))   # (1, 2)
        return {"wsi_logits": wsi_logits, "attn_weights": attn_weights}
