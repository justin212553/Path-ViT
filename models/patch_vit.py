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

    [존재 이유]
    ViT를 지난 뒤 N개의 패치 토큰이 남는다. 이 토큰들은 WSI 내 각 위치의 표현이지만,
    최종 분류는 WSI 단위 단일 벡터를 요구한다.
    ABMIL은 "어떤 패치가 WSI 라벨 결정에 중요한가"를 학습 가능한 attention 가중치로
    결정해 N개 토큰을 1개 WSI 임베딩으로 집계한다.

    [Cluster Query Token으로 전환 시 이 모듈이 제거되는 이유]
    Cluster Query Token 방식에서는 K개의 쿼리 토큰이 ViT 내부 attention을 통해
    이미 유형별 집계를 완료한다. 즉 "N → 1 집계" 문제가 "K개 유형별 표현 → 히스토그램
    가중합"으로 대체되므로, 별도의 ABMIL 단계가 불필요해진다.

    [구조]
    attn_v: tanh 게이트  — 패치 표현의 방향성 포착
    attn_u: sigmoid 게이트 — 패치 표현의 크기/활성 포착
    두 게이트의 element-wise 곱 → attn_w로 스칼라 점수 산출 (gated attention)
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attn_v = nn.Linear(embed_dim, hidden_dim)   # tanh 게이트
        self.attn_u = nn.Linear(embed_dim, hidden_dim)   # sigmoid 게이트
        self.attn_w = nn.Linear(hidden_dim, 1)           # 스칼라 점수

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (N, D) — ViT를 지난 패치 토큰. N은 WSI마다 다름.
        Returns:
            wsi_embed:    (D,) — attention 가중합으로 집계된 WSI 임베딩
            attn_weights: (N,) — 패치별 attention 가중치 (합=1, 시각화·해석용)
        """
        # gated attention: tanh × sigmoid → 두 게이트가 방향성과 크기를 동시에 제어
        gate = torch.tanh(self.attn_v(tokens)) * torch.sigmoid(self.attn_u(tokens))  # (N, H)

        # 각 패치의 중요도 점수 → softmax로 확률 분포화 (합=1 보장)
        scores = self.attn_w(gate).squeeze(-1)        # (N,)
        attn_weights = torch.softmax(scores, dim=0)   # (N,)

        # 중요도 가중합으로 N개 패치 토큰을 단일 WSI 임베딩으로 집계
        wsi_embed = (attn_weights.unsqueeze(-1) * tokens).sum(dim=0)  # (D,)
        return wsi_embed, attn_weights


class PatchViT(nn.Module):
    def __init__(self, cfg: ModelConfig, precomputed: bool = True):
        """
        Args:
            precomputed: True면 ResNet50 backbone을 생성하지 않는다 — 항상 사전 추출된
                         pooled feature(features 인자)만 입력으로 받는 모드.
                         False면 patch_paths로 이미지를 직접 디코딩/forward한다.
        """
        super().__init__()
        self.precomputed = precomputed
        self.cnn = CNNEncoder(cfg.embed_dim, with_backbone=not precomputed)
        self.vit = ViTEncoder(cfg.embed_dim, cfg.num_heads,
                              cfg.num_transformer_layers, cfg.dropout,
                              cfg.max_grid_size, use_grad_checkpoint=cfg.grad_checkpoint,
                              num_landmarks=cfg.num_landmarks)
        self.attn_pool = AttentionPooling(cfg.embed_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, 2),
        )

    def forward(
        self,
        coords: torch.Tensor,
        patch_paths: list[Path] | None = None,
        features: torch.Tensor | None = None,
        transform=None,
        chunk_size: int | None = None,
    ) -> dict:
        """
        Args:
            coords:      (N_patches, 2)
            patch_paths: N개 패치 이미지 파일 경로 (precomputed=False 모드) — 이미지 디코딩을
                         chunk_size 단위로 지연 로딩해 한 번에 메모리에 올리는 패치 수를 제한한다
                         (패치 수에 cap이 없는 대형 WSI에서 host RAM OOM 방지)
            features:    (N_patches, 2048) 사전 추출된 backbone+pool feature (precomputed=True 모드)
            transform:   패치 이미지 → 텐서 변환 (CAMELYON17NodeDataset.transform). patch_paths 모드에서만 사용
            chunk_size:  CNN을 이 크기 단위로 나눠 실행. None이면 한 번에 실행. patch_paths 모드에서만 사용
        Returns:
            wsi_logits:   (1, 2)
            attn_weights: (N_patches,)
        """
        device = coords.device

        if features is not None:
            patch_tokens = self.cnn.forward_pooled(features.to(device, non_blocking=True))
        else:
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
