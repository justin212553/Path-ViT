"""비학습(non-parametric) 공간 특징 — kNN/hybrid attention 계열이 PAAD(pre-augment, PMA/M4A
둘 다)에서 전부 baseline을 못 넘은 뒤(2026-07-23), "새로 학습되는 attention 파라미터 자체가
작은 코호트에서 과적합 유인 아니냐"는 가설을 검증하기 위한 대안이다. 좌표/패치임베딩/attention
가중치로부터 결정론적으로 계산되는 스칼라 몇 개만 risk_head 입력에 추가한다 — 이 계산 자체엔
학습 파라미터가 전혀 없고(risk_head의 확장된 입력 차원에 대응하는 몇 개의 새 가중치만 학습됨),
gradient는 이 경로를 통해 patch_tokens/attn_weights까지 흘러가지만(detach 안 함) 새 attention
모듈처럼 자체 q/k/v projection을 갖지 않는다.
"""
import torch
import torch.nn.functional as F

from .spatial_residual import knn_edges


def spatial_autocorr(patch_tokens: torch.Tensor, coords: torch.Tensor, k: int = 8) -> torch.Tensor:
    """패치 임베딩의 지역적 동질성(local homogeneity) — 각 패치와 kNN 이웃들의 코사인 유사도를
    평균/표준편차로 요약한다(Moran's I류 공간 자기상관 통계의 임베딩 버전). "조직이 공간적으로
    얼마나 균질/이질적인가"를 attention 없이 직접 수치화 — 학습 파라미터 없음.

    Returns: (2,) — [mean, std] of per-patch mean-neighbor-cosine-similarity.
    """
    n = patch_tokens.shape[0]
    if n < 2:
        return torch.zeros(2, device=patch_tokens.device, dtype=torch.float32)
    edge_index = knn_edges(coords, k)
    if edge_index.shape[1] == 0:
        return torch.zeros(2, device=patch_tokens.device, dtype=torch.float32)
    src, dst = edge_index[0], edge_index[1]
    sims = F.cosine_similarity(patch_tokens[src].float(), patch_tokens[dst].float(), dim=-1)  # (E,)
    per_node_sum = torch.zeros(n, device=patch_tokens.device, dtype=torch.float32)
    deg = torch.zeros(n, device=patch_tokens.device, dtype=torch.float32)
    per_node_sum.index_add_(0, src, sims)
    deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    per_node_mean = per_node_sum / deg.clamp(min=1)
    std = per_node_mean.std(unbiased=False) if n > 1 else torch.zeros((), device=patch_tokens.device)
    return torch.stack([per_node_mean.mean(), std])


def attention_dispersion(coords: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
    """attention이 "중요하다"고 뽑은 패치들이 공간적으로 뭉쳐있는지(낮음) 흩어져있는지(높음) —
    attn_weights(합=1)로 가중한 좌표의 표준편차(등방 근사, row/col 평균). 침윤 전선처럼 국소적으로
    뭉친 패턴인지, 슬라이드 전역에 퍼진 패턴인지를 직접 수치화 — 학습 파라미터 없음.

    Returns: (1,) — weighted spatial std.
    """
    coords_f = coords.float()
    w = attn_weights.float()
    w = w / w.sum().clamp(min=1e-8)
    centroid = (w.unsqueeze(-1) * coords_f).sum(dim=0)                # (2,)
    diff = coords_f - centroid                                        # (N, 2)
    weighted_var = (w.unsqueeze(-1) * diff.pow(2)).sum(dim=0)          # (2,) row/col 분산
    weighted_std = weighted_var.clamp(min=0).sqrt().mean()             # 스칼라(등방 근사)
    return weighted_std.view(1)
