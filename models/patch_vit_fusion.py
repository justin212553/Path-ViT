"""
LateFusionViT — ViT+ABMIL과 Cluster Histogram Branch의 Late Fusion 모델

[두 경로가 포착하는 정보]
  Path A (ViT+ABMIL)        : 패치 간 공간 배열과 문맥 — "종양 패치가 어떻게 배열되어 있는가"
  Path B (ClusterHistogram) : WSI 조직 구성 비율     — "어떤 조직 유형이 얼마나 존재하는가"

두 경로는 같은 features.pt를 입력으로 받되 서로 다른 정보를 추출하므로 상보적이다.
Path B의 k-means 군집 중심(centroids)은 학습 전 사전 계산되며 학습 중 고정된다.

[이 모델의 위치]
  PatchViT (ViT+ABMIL)
      → [현재] LateFusionViT (ViT+ABMIL + Cluster Histogram, Late Fusion)
      → [다음]  ClusterQueryViT (ABMIL 제거, K개 query token이 ViT 내부에서 집계)

  Late Fusion 단계에서 Path B의 성능 기여를 ablation으로 확인한 뒤,
  기여가 유효하면 Cluster Query Token 방식으로 전환한다.

[k-means 사전 계산]
  학습 전 data/fit_clusters.py 를 실행해 cluster_centroids.pt (K, 2048) 를 생성해야 한다.
  LateFusionViT 생성 시 해당 텐서를 cluster_centroids 인자로 전달한다.

Forward 출력:
    wsi_logits   : (1, 2)       — WSI 단위 이진 분류 logit (정상 / 전이)
    attn_weights : (N_patches,) — ABMIL 패치 attention 가중치 (시각화·해석용)
    histogram    : (K,)         — 군집별 패치 비율 (해석용)
"""
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image

from .cnn_encoder import CNNEncoder
from .patch_vit import AttentionPooling
from .vit_encoder import ViTEncoder
from config import ModelConfig


class ClusterHistogramBranch(nn.Module):
    """
    Path B: raw CNN feature (N, 2048) → k-means 할당 → 비율 히스토그램 (K,) → WSI 임베딩 (D,)

    [역할]
    ViT+ABMIL이 패치 순서와 공간 배열에 의존하는 반면, 이 브랜치는
    WSI 전체의 조직 구성 비율만 본다. 히스토그램은 패치 순서에 무관
    (permutation-invariant)하므로 Path A와 정보가 겹치지 않는다.

    [학습 범위]
    centroids (k-means 중심) : 고정 — 군집 구조는 unsupervised로 사전 결정됨
    hist_mlp                 : 학습 — 비율 히스토그램을 분류에 유용한 임베딩으로 변환
    """

    def __init__(self, num_clusters: int, embed_dim: int):
        super().__init__()
        # 히스토그램 (K,) → 임베딩 (D,): 두 층 MLP로 비선형 변환
        self.hist_mlp = nn.Sequential(
            nn.Linear(num_clusters, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def compute_histogram(
        self, raw_features: torch.Tensor, centroids: torch.Tensor
    ) -> torch.Tensor:
        """
        각 패치를 최근접 군집 중심에 할당한 뒤 군집별 패치 비율을 계산한다.

        Args:
            raw_features : (N, 2048) backbone+pool 출력 (proj 이전의 원본 feature)
            centroids    : (K, 2048) k-means 군집 중심 (frozen buffer)
        Returns:
            histogram    : (K,)  각 군집에 속하는 패치 비율 (합=1)
        """
        K = centroids.shape[0]
        # 유클리드 거리로 nearest centroid 할당 — bfloat16 정밀도 손실 방지를 위해 float32 강제
        dists = torch.cdist(raw_features.float(), centroids.float())  # (N, K)
        assignments = dists.argmin(dim=-1)                            # (N,)
        counts = torch.bincount(assignments, minlength=K).float()     # (K,)
        return counts / counts.sum()                                  # (K,) 비율 정규화

    def forward(
        self, raw_features: torch.Tensor, centroids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_hist    : (1, D) — 히스토그램을 변환한 WSI 임베딩
            histogram : (K,)   — 원본 비율 벡터 (해석·로깅용)
        """
        histogram = self.compute_histogram(raw_features, centroids)  # (K,)
        z_hist = self.hist_mlp(histogram.unsqueeze(0))               # (1, D)
        return z_hist, histogram


class LateFusionViT(nn.Module):
    """
    ViT+ABMIL (Path A) + Cluster Histogram Branch (Path B) Late Fusion.

    [Fusion 구조]
      z_vit  (1, D) — Path A 출력: 공간 문맥이 반영된 WSI 임베딩
      z_hist (1, D) — Path B 출력: 조직 구성 비율이 반영된 WSI 임베딩
        → concat → (1, 2D) → LayerNorm → Linear → logits (1, 2)

    [Cluster Query Token으로의 전환 경로]
      이 모델에서 Path B가 성능에 기여함이 확인되면 다음 단계로 이동한다:
        1. attn_pool (ABMIL) 제거
        2. ClusterHistogramBranch 제거
        3. ViT 입력에 K개 cluster query token 추가
        4. ViT 출력의 K query token × histogram 가중합 → WSI 임베딩 → 분류기
      이 전환 후 classifier 입력 차원이 2D → D로 줄어든다.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        cluster_centroids: torch.Tensor,
        precomputed: bool = True,
    ):
        """
        Args:
            cfg               : ModelConfig
            cluster_centroids : (K, 2048) 사전 계산된 k-means 군집 중심. 학습 중 고정.
            precomputed       : True면 CNN backbone 없이 features.pt 사용
        """
        super().__init__()
        K = cluster_centroids.shape[0]

        # k-means 중심: gradient 없이 device 이동만 지원하는 buffer로 등록
        self.register_buffer("centroids", cluster_centroids.float())  # (K, 2048)

        # Path A: ViT+ABMIL (공간 문맥)
        self.precomputed = precomputed
        self.cnn = CNNEncoder(cfg.embed_dim, with_backbone=not precomputed)
        self.vit = ViTEncoder(
            cfg.embed_dim, cfg.num_heads,
            cfg.num_transformer_layers, cfg.dropout,
            use_grad_checkpoint=cfg.grad_checkpoint,
            num_landmarks=cfg.num_landmarks,
        )
        self.attn_pool = AttentionPooling(cfg.embed_dim)

        # Path B: 클러스터 히스토그램 (조직 구성 비율)
        self.hist_branch = ClusterHistogramBranch(K, cfg.embed_dim)

        # Late Fusion 분류기: [z_vit ‖ z_hist] (2D,) → logits (2,)
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim * 2),
            nn.Linear(cfg.embed_dim * 2, 2),
        )

    def _extract_raw_features(
        self,
        patch_paths: list[Path],
        transform,
        chunk_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        raw image 모드 전용: backbone+pool까지 실행해 (N, 2048) raw feature를 추출한다.
        proj는 적용하지 않는다 — histogram 계산에 2048-dim 원본이 필요하기 때문.
        """
        chunks = []
        for i in range(0, len(patch_paths), chunk_size):
            batch = torch.stack([
                transform(Image.open(p).convert("RGB"))
                for p in patch_paths[i: i + chunk_size]
            ]).to(device, non_blocking=True)
            feat_map = self.cnn.backbone(batch)
            pooled = self.cnn.pool(feat_map)   # (B, 2048)
            chunks.append(pooled)
        return torch.cat(chunks)               # (N, 2048)

    def forward(
        self,
        coords: torch.Tensor,
        patch_paths: Optional[list[Path]] = None,
        features: Optional[torch.Tensor] = None,
        transform=None,
        chunk_size: Optional[int] = None,
    ) -> dict:
        """
        Args:
            coords      : (N, 2)
            features    : (N, 2048) 사전 추출 feature — precomputed=True 모드
            patch_paths : 패치 이미지 경로 리스트   — precomputed=False 모드
            transform   : 이미지 전처리 변환         — patch_paths 모드에서만 사용
            chunk_size  : CNN chunk 크기             — patch_paths 모드에서만 사용
        Returns:
            wsi_logits   : (1, 2)  — 이진 분류 logit
            attn_weights : (N,)    — ABMIL 패치 attention 가중치 (시각화용)
            histogram    : (K,)    — 군집별 패치 비율 (해석용)
        """
        device = coords.device

        if features is not None:
            # precomputed 모드: features.pt의 2048-dim feature를 그대로 histogram에도 사용
            raw_features = features.to(device, non_blocking=True)   # (N, 2048)
            patch_tokens = self.cnn.forward_pooled(raw_features)    # (N, D)
        else:
            chunk_size = chunk_size or len(patch_paths)
            raw_features = self._extract_raw_features(              # (N, 2048)
                patch_paths, transform, chunk_size, device
            )
            patch_tokens = self.cnn.forward_pooled(raw_features)    # (N, D)

        # Path A: 공간 문맥 → ABMIL 집계
        ctx_tokens = self.vit(patch_tokens, coords)                 # (N, D)
        z_vit, attn_weights = self.attn_pool(ctx_tokens)            # (D,), (N,)

        # Path B: 조직 구성 비율 → 히스토그램 임베딩
        z_hist, histogram = self.hist_branch(raw_features, self.centroids)  # (1, D), (K,)

        # Late Fusion: z_vit와 z_hist를 concat 후 분류
        z_fused = torch.cat([z_vit.unsqueeze(0), z_hist], dim=-1)  # (1, 2D)
        wsi_logits = self.classifier(z_fused)                       # (1, 2)

        return {
            "wsi_logits":   wsi_logits,
            "attn_weights": attn_weights,
            "histogram":    histogram,
        }
