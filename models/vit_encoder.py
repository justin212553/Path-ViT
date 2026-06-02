"""
ViT Encoder with 2D Spatial Position Embedding
- 림프절 내부 해부학적 구조 보존을 위한 2D 공간적 위치 인코딩
- Transformer Encoder로 Global Spatial Context 연산
- 출력: 정제된 이미지 토큰 H_img
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
    패치 토큰 + 공간 임베딩 → Transformer Encoder → H_img
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

        # [CLS] 토큰: 슬라이드 전체를 대표하는 글로벌 표현
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self, patch_tokens: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: (N_patches, embed_dim) - CNN에서 추출한 패치 특징
            coords:       (N_patches, 2)          - 각 패치의 (row, col) 좌표
        Returns:
            h_img:     (1, embed_dim)      - 슬라이드 전체 표현 ([CLS] 토큰)
            all_tokens:(N_patches+1, embed_dim) - 모든 패치 토큰 (Heatmap 생성용)
        """
        # 공간 임베딩 주입
        pos = self.pos_embedding(coords)           # (N, D)
        x = patch_tokens + pos                     # (N, D)

        # [CLS] 토큰 prepend → (1, N+1, D)
        cls = self.cls_token.expand(1, -1, -1)
        x = torch.cat([cls, x.unsqueeze(0)], dim=1)  # (1, N+1, D)

        # Transformer 연산 (학습 시 gradient checkpointing으로 활성화 메모리 절감)
        if self.use_grad_checkpoint and self.training:
            out = checkpoint_sequential(self.transformer.layers, len(self.transformer.layers), x, use_reentrant=False)
            if self.transformer.norm is not None:
                out = self.transformer.norm(out)
        else:
            out = self.transformer(x)              # (1, N+1, D)

        h_img = out[:, 0, :]                       # (1, D) - CLS 토큰
        all_tokens = out[0]                        # (N+1, D)

        return h_img, all_tokens
