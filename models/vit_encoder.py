"""
ViT Encoder with 2D Spatial Position Embedding
- 림프절 내부 해부학적 구조 보존을 위한 2D 공간적 위치 인코딩
- Transformer Encoder로 Global Spatial Context 연산
- 출력: 공간 문맥이 반영된 패치 토큰 (N, D)
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential


class SpatialPositionEmbedding(nn.Module):
    """
    WSI 패치의 (row, col) 그리드 좌표를 embed_dim 벡터로 인코딩.
    학습 가능한 embedding을 사용하여 림프절 내부 구조적 위치 반영.
    """

    def __init__(self, max_grid_size: int = 64, embed_dim: int = 512):
        super().__init__()
        self.row_embed = nn.Embedding(max_grid_size, embed_dim // 2)
        self.col_embed = nn.Embedding(max_grid_size, embed_dim // 2)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N_patches, 2) - 각 패치의 (row, col) 그리드 좌표
        Returns:
            pos_embed: (N_patches, embed_dim)
        """
        rows = self.row_embed(coords[:, 0])  # (N, D/2)
        cols = self.col_embed(coords[:, 1])  # (N, D/2)
        return torch.cat([rows, cols], dim=-1)  # (N, D)


class ViTEncoder(nn.Module):
    """
    패치 토큰 + 공간 임베딩 → Transformer Encoder → 문맥화된 패치 토큰
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_grid_size: int = 64,
        use_grad_checkpoint: bool = True,
    ):
        super().__init__()
        self.pos_embedding = SpatialPositionEmbedding(max_grid_size, embed_dim)
        self.use_grad_checkpoint = use_grad_checkpoint

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,   # (B, N, D) 형식 사용
            norm_first=True,    # Pre-LN: 학습 안정성 향상
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self, patch_tokens: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens: (N_patches, embed_dim) - CNN에서 추출한 패치 특징
            coords:       (N_patches, 2)          - 각 패치의 (row, col) 좌표
        Returns:
            out_tokens: (N_patches, embed_dim) - 공간 문맥이 반영된 패치 토큰
        """
        pos = self.pos_embedding(coords)       # (N, D)
        x = (patch_tokens + pos).unsqueeze(0)  # (1, N, D)

        if self.use_grad_checkpoint and self.training:
            out = checkpoint_sequential(self.transformer.layers, len(self.transformer.layers), x, use_reentrant=False)
            if self.transformer.norm is not None:
                out = self.transformer.norm(out)
        else:
            out = self.transformer(x)          # (1, N, D)

        return out[0]                          # (N, D)
