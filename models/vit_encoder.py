"""
ViT Encoder with 2D Spatial Position Embedding
- 림프절 내부 해부학적 구조 보존을 위한 2D 공간적 위치 인코딩
- Nystrom 근사 self-attention으로 Global Spatial Context 연산 (O(N) 복잡도, 대규모 패치 수 대응)
- 출력: 공간 문맥이 반영된 패치 토큰 (N, D)
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
from nystrom_attention import NystromAttention


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


class NystromEncoderLayer(nn.Module):
    """
    Pre-LN Transformer 블록, self-attention만 Nystrom 근사로 교체.
    전체 softmax attention(O(N^2)) 대신 landmark 기반 근사(O(N))를 사용해
    패치 수가 매우 큰 WSI(수만 패치)에서도 attention 연산이 가능하게 한다.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_landmarks: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = NystromAttention(
            dim=embed_dim,
            dim_head=embed_dim // num_heads,
            heads=num_heads,
            num_landmarks=num_landmarks,
            pinv_iterations=6,
            residual=True,
            dropout=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    패치 토큰 + 공간 임베딩 → Nystrom Transformer Encoder → 문맥화된 패치 토큰
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_grid_size: int = 64,
        use_grad_checkpoint: bool = True,
        num_landmarks: int = 128,
    ):
        super().__init__()
        self.pos_embedding = SpatialPositionEmbedding(max_grid_size, embed_dim)
        self.use_grad_checkpoint = use_grad_checkpoint

        self.layers = nn.ModuleList([
            NystromEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                num_landmarks=num_landmarks,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

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
            out = checkpoint_sequential(self.layers, len(self.layers), x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x)
            out = x                            # (1, N, D)

        return self.norm(out[0])               # (N, D)
