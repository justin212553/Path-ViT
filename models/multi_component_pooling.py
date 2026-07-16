"""
MultiComponentPooling — ABMIL의 단일 벡터 압축을 피하기 위해, 여러 통계적 관점(mean, std,
attention-weighted, top-k-mean)을 압축 없이 병렬로 유지하는 pooling 모듈.

배경(findings_backlog.md 1번 항목): --dataset both 비교에서 우리 M4(0.539)가 레퍼런스
M4(0.722)에 크게 못 미쳤는데, 우리 M1은 레퍼런스 M1과 거의 일치했다 — 즉 병리 인코더/MIL
집계 자체가 아니라 "WSI를 단일 벡터로 압축한 뒤 그 위에 RNA를 개입시키는" fusion 설계가
병목이라는 뜻이다. 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology)의 Morphology Burden
Pooling을 단순화해서 이식한다 — 원본은 mean/std/risk-weighted/top10%/top25%/risk 분포
통계(10개)까지 총 6그룹을 쓰지만, 여기서는 핵심 4개(mean/std/attention-weighted/
top-k-mean)만으로 먼저 시작한다(분포 통계 그룹은 필요하면 나중에 확장).

attention-weighted 성분은 기존 AttentionPooling(ABMIL, vit_m1.py)을 그대로 재사용한다 —
이 프로젝트가 이미 검증한 gated-attention 메커니즘을 "여러 관점 중 하나"로 편입시키는
설계다(완전히 새로 만들지 않고 기존에 검증된 요소를 보존).
"""
import torch
import torch.nn as nn

from .vit_m1 import AttentionPooling


class MultiComponentPooling(nn.Module):
    """
    N개 패치 토큰 → (mean, std, attention-weighted, top-k-mean) 4개 관점 (K, D).
    단일 벡터로 합치지 않고 성분을 그대로 유지 — 소비하는 쪽(PM4는 flatten+게이트,
    PMA는 co-attention)이 각자의 방식으로 결합한다.
    """

    NUM_COMPONENTS = 4

    def __init__(self, embed_dim: int, hidden_dim: int = 128, top_frac: float = 0.1):
        super().__init__()
        self.attn = AttentionPooling(embed_dim, hidden_dim=hidden_dim)
        self.top_frac = top_frac

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (N, D) — ViT를 지난 패치 토큰
        Returns:
            components:   (4, D) — [mean, std, attention-weighted, top-k-mean]
            attn_weights: (N,)   — attention-weighted 성분 계산에 쓰인 가중치 (시각화/해석용)
        """
        n = tokens.shape[0]
        h_mean = tokens.mean(dim=0)
        h_std = tokens.std(dim=0, unbiased=False) if n > 1 else torch.zeros_like(h_mean)
        h_attn, attn_weights = self.attn(tokens)  # (D,), (N,)

        k = max(1, round(n * self.top_frac))
        top_idx = torch.topk(attn_weights, k).indices
        h_top = tokens[top_idx].mean(dim=0)

        components = torch.stack([h_mean, h_std, h_attn, h_top], dim=0)  # (4, D)
        return components, attn_weights
