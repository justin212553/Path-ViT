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
    WSI 패치의 (row, col) 좌표를 sinusoidal 인코딩으로 변환.

    학습 파라미터 없음 — 좌표값 자체를 결정론적으로 인코딩.
    embed_dim은 4의 배수여야 한다 (row·col 각각 sin/cos 절반씩 할당).
    """

    def __init__(self, embed_dim: int = 512, temperature: float = 10000.0):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        self.embed_dim  = embed_dim
        self.temperature = temperature

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N_patches, 2) int64 — 각 패치의 (row, col) 그리드 좌표
        Returns:
            pos_embed: (N_patches, embed_dim)
        """
        quarter = self.embed_dim // 4
        freq = self.temperature ** (
            torch.arange(quarter, device=coords.device, dtype=torch.float32) / quarter
        )                                               # (D/4,)  주파수 분모
        row = coords[:, 0:1].float() / freq             # (N, D/4)
        col = coords[:, 1:2].float() / freq             # (N, D/4)
        return torch.cat(
            [row.sin(), row.cos(), col.sin(), col.cos()], dim=-1
        )                                               # (N, D)


class NystromEncoderLayer(nn.Module):
    """
    Pre-LN Transformer 블록, self-attention만 Nystrom 근사로 교체.
    전체 softmax attention(O(N^2)) 대신 landmark 기반 근사(O(N))를 사용해
    패치 수가 매우 큰 WSI(수만 패치)에서도 attention 연산이 가능하게 한다.

    [use_ffn=False, --M4A_FF 맛보기 ablation] FFN 서브레이어를 통째로 제거한다.
    attention은 패치 간 정보를 "섞는" 역할, FFN은 그렇게 섞인 결과를 패치 하나 단위로
    비선형 변환하는 역할로 서로 독립적이다(FFN은 다른 패치를 전혀 참조하지 않음) — 그래서
    FFN을 빼도 공간 컨텍스트(attention이 만든) 자체는 그대로 남고, "그 결과를 더 다듬는"
    단계만 없어진다.

    [context_dim, --M2_FF 맛보기 ablation] FFN 입력 직전에 외부 컨텍스트(z_rna)를 가산
    bias로 더한다(vit_m1.py::AttentionPooling의 FiLM 방식과 동일한 관례) — "attention이
    이미 공간 문맥을 반영한 패치 표현"을 RNA 기준으로 한 번 더 조건화한 뒤 FFN으로 다듬는
    지점. use_ffn=True일 때만 의미가 있다(FFN이 없으면 조건화할 대상이 없음).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_landmarks: int,
        use_ffn: bool = True,
        context_dim: int | None = None,
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

        self.use_ffn = use_ffn
        self.context_proj: nn.Linear | None = None
        if use_ffn:
            self.norm2 = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, embed_dim),
                nn.Dropout(dropout),
            )
            if context_dim is not None:
                self.context_proj = nn.Linear(context_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.norm1(x)))
        if self.use_ffn:
            h = self.norm2(x)
            if self.context_proj is not None and context is not None:
                h = h + self.context_proj(context)  # (1, N, D) + (D,) broadcast
            x = x + self.ffn(h)
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
        use_grad_checkpoint: bool = True,
        num_landmarks: int = 128,
        use_ffn: bool = True,
        context_dim: int | None = None,
    ):
        super().__init__()
        self.pos_embedding = SpatialPositionEmbedding(embed_dim)
        self.use_grad_checkpoint = use_grad_checkpoint

        self.layers = nn.ModuleList([
            NystromEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                num_landmarks=num_landmarks,
                use_ffn=use_ffn,
                context_dim=context_dim,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, patch_tokens: torch.Tensor, coords: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens: (N_patches, embed_dim) - CNN에서 추출한 패치 특징
            coords:       (N_patches, 2)          - 각 패치의 (row, col) 좌표
            context:      (embed_dim,) 선택 — 각 레이어의 FFN 직전에 가산 bias로 주입할
                          외부 컨텍스트(z_rna). --M2_FF 맛보기 ablation 전용, 기본 None이면
                          기존 모델들과 완전히 동일하게 동작한다.
        Returns:
            out_tokens: (N_patches, embed_dim) - 공간 문맥이 반영된 패치 토큰
        """
        pos = self.pos_embedding(coords)       # (N, D)
        x = (patch_tokens + pos).unsqueeze(0)  # (1, N, D)

        if context is None and self.use_grad_checkpoint and self.training:
            out = checkpoint_sequential(self.layers, len(self.layers), x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x, context=context)
            out = x                            # (1, N, D)

        return self.norm(out[0])               # (N, D)
