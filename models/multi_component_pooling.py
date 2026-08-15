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

2026-08-14: diagnose_pma_wsi_structure.py 실측 — self.attn(ABMIL)의 patch attention entropy가
~0.999(사실상 uniform)로 붕괴해 있었고, attn 자신의 gradient norm도 다른 모듈 대비
100~250배 작았다. 그런데 기존 "top" 컴포넌트는 이 attn_weights로 top-k를 선정하고 있었다 —
독립적인 관점이 아니라 이미 무너진 attention에 얹혀 있었던 것(top-k를 지워도 external이
오히려 소폭 긍정적이었던 과거 관찰과 부합). 레퍼런스 MorphologyBurdenPooling
(scripts/models/morphology_burden_mil.py)을 참고해, top-k 선정을 attn_pool과 파라미터를
공유하지 않는 별도의 단순 TileRiskHead(게이트 없는 얕은 MLP)로 분리할 수 있게 했다
(use_tile_risk_head=True, 기본 False로 기존 동작 완전히 유지). 켜면 레퍼런스의
risk_stats(패치별 risk 점수 분포를 요약하는 10개 스칼라)도 함께 계산해 반환한다.
"""
import torch
import torch.nn as nn

from .vit_m1 import AttentionPooling

ALL_COMPONENTS = ("mean", "std", "attn", "top")
NUM_RISK_STATS = 10


class TileRiskHead(nn.Module):
    """게이트 없는 얕은 MLP로 패치별 독립적인 risk 점수를 계산한다(레퍼런스
    MorphologyBurdenPooling.tile_risk_head와 동일 구조) — self.attn(ABMIL)과 파라미터를 전혀
    공유하지 않아, attn의 붕괴(entropy~0.999, 2026-08-14 실측)를 물려받지 않는다."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (N, D) -> raw_risk: (N,)"""
        return self.net(tokens).squeeze(-1)


def _topk_mean(values: torch.Tensor, scores: torch.Tensor, fraction: float) -> torch.Tensor:
    """scores 상위 fraction 비율의 인덱스로 values를 골라 평균한다 — values가 (N, D) 패치
    토큰이든 (N,) 스칼라(tile_risk 자기 자신)든 그대로 재사용된다."""
    n = scores.shape[0]
    k = max(1, int(torch.ceil(torch.tensor(n * fraction, device=scores.device)).item()))
    k = min(k, n)
    idx = torch.topk(scores, k=k, largest=True).indices
    return values.index_select(0, idx).mean(dim=0)


class MultiComponentPooling(nn.Module):
    """
    N개 패치 토큰 → (mean, std, attention-weighted, top-k-mean) 4개 관점 (K, D).
    단일 벡터로 합치지 않고 성분을 그대로 유지 — 소비하는 쪽(PM4는 flatten+게이트,
    PMA는 co-attention)이 각자의 방식으로 결합한다.

    2026-08-09: scripts/diagnose_pma_component_reliance.py의 사후 zero-ablation에서 4개
    성분 중 어느 것을 지워도 손해가 아니었던(오히려 top-k 제거가 external에 소폭 긍정적)
    관찰을 이어받아, 구조적으로 하나를 아예 빼고 처음부터 재학습해볼 수 있게 `exclude`를
    추가했다 — co-attention/self-attention은 입력 토큰 개수(K)에 무관하게 동작하므로(vit_m4a.py
    ::CoAttentionPooling, vit_m1_pool.py::SelfAttentionPooling 둘 다 일반적인 N-토큰 attention),
    이 변경이 risk_head 등 다른 차원에 영향을 주지 않는다. `NUM_COMPONENTS`(PM4가 참조하는 클래스
    상수)는 건드리지 않는다 — exclude는 PMA 계열 ablation 전용이라 PM4의 flatten 차원 계산과는
    무관해야 한다.

    forward()는 항상 (components, attn_weights, risk_stats) 3-tuple을 반환한다 — risk_stats는
    use_tile_risk_head=False(기본)면 None.
    """

    NUM_COMPONENTS = 4

    def __init__(self, embed_dim: int, hidden_dim: int = 128, top_frac: float = 0.1,
                 exclude: str | None = None, use_tile_risk_head: bool = False,
                 risk_head_dropout: float = 0.3):
        super().__init__()
        if exclude is not None and exclude not in ALL_COMPONENTS:
            raise ValueError(f"알 수 없는 exclude: {exclude!r} (선택지: {ALL_COMPONENTS})")
        self.attn = AttentionPooling(embed_dim, hidden_dim=hidden_dim)
        self.top_frac = top_frac
        self.exclude = exclude
        self.use_tile_risk_head = use_tile_risk_head
        if use_tile_risk_head:
            self.tile_risk_head = TileRiskHead(embed_dim, dropout=risk_head_dropout)

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: (N, D) — ViT를 지난 패치 토큰
        Returns:
            components:   (4, D) 또는 exclude가 있으면 (3, D) — [mean, std, attention-weighted, top-k-mean]
            attn_weights: (N,)   — attention-weighted 성분 계산에 쓰인 가중치 (시각화/해석용).
                          exclude="attn"이어도 계산은 항상 한다(top-k가 예전엔 이걸 썼음).
            risk_stats:   (10,) 또는 None — use_tile_risk_head=True일 때만, TileRiskHead 점수
                          분포 요약(레퍼런스 MorphologyBurdenPooling과 동일 정의).
        """
        n = tokens.shape[0]
        h_mean = tokens.mean(dim=0)
        h_std = tokens.std(dim=0, unbiased=False) if n > 1 else torch.zeros_like(h_mean)
        h_attn, attn_weights = self.attn(tokens)  # (D,), (N,)

        risk_stats = None
        if self.use_tile_risk_head:
            # 2026-08-14: top-k 선정 기준을 attn_weights(붕괴됨)에서 독립적인 tile_risk_head로 분리.
            # .float(): bf16 autocast 하에서 torch.quantile()이 bfloat16을 지원하지 않아
            # RuntimeError가 남 — attention_dispersion()/spatial_autocorr()와 동일한 이유로 float32 승격.
            raw_risk = self.tile_risk_head(tokens).float()  # (N,)
            tile_risk = torch.sigmoid(raw_risk)              # (N,) — 0~1, risk_stats 산출용
            h_top = _topk_mean(tokens, tile_risk, self.top_frac)
            risk_stats = self._risk_stats(tile_risk)
        else:
            k = max(1, round(n * self.top_frac))
            top_idx = torch.topk(attn_weights, k).indices
            h_top = tokens[top_idx].mean(dim=0)

        parts = []
        if self.exclude != "mean": parts.append(h_mean)
        if self.exclude != "std": parts.append(h_std)
        if self.exclude != "attn": parts.append(h_attn)
        if self.exclude != "top": parts.append(h_top)
        components = torch.stack(parts, dim=0)  # (4 또는 3, D)
        return components, attn_weights, risk_stats

    @staticmethod
    def _risk_stats(tile_risk: torch.Tensor) -> torch.Tensor:
        """레퍼런스 MorphologyBurdenPooling.forward()와 동일한 10개 스칼라 — tile_risk(패치별
        0~1 점수) 분포 요약: mean, std, max, q25, q50, q75, top05_mean, top10_mean,
        frac_over_50(연성 임계값), frac_over_70. "이 슬라이드에서 고위험으로 보이는 패치가
        얼마나 되는가"를 pooled feature 벡터가 아니라 risk 점수 자체의 분포로 직접 표현한다."""
        q25 = torch.quantile(tile_risk, 0.25)
        q50 = torch.quantile(tile_risk, 0.50)
        q75 = torch.quantile(tile_risk, 0.75)
        top05 = _topk_mean(tile_risk, tile_risk, 0.05)
        top10 = _topk_mean(tile_risk, tile_risk, 0.10)
        frac_over_50 = torch.sigmoid((tile_risk - 0.50) * 20.0).mean()
        frac_over_70 = torch.sigmoid((tile_risk - 0.70) * 20.0).mean()
        return torch.stack([
            tile_risk.mean(), tile_risk.std(unbiased=False), tile_risk.max(),
            q25, q50, q75, top05, top10, frac_over_50, frac_over_70,
        ])
