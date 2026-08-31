"""
BilinearFusion — PORPOISE(Chen et al. 2022, Cancer Cell)/Pathomic Fusion(Chen et al. 2020)의
Kronecker Product Fusion을 재현. models/vit_porpoise.py::ViT_PORPOISE(--PORPOISE)에서 사용.

2026-08-31 배경: findings_backlog.md 최상위 발견(MCAT/M4A/PMA 전부 co-attention entropy가
0.999~1.000으로 붕괴 — query 개수를 1개→8개로 늘려도 재발, gradient는 정상 도달하는데도
attention이 patch를 구별 못 함)에 따라, "attention이 중요한 patch를 찾아내야 하는" 구조
계열(M4/M4A/PMA/MCAT 전부 이 계열) 자체를 벗어나기 위한 Phase 3. PORPOISE는 WSI를 평범한
(RNA-무관) gated-ABMIL로 풀링하고, RNA와의 상호작용은 풀링 *이후* 명시적 pairwise interaction
(Kronecker/outer product)으로 포착한다 — "어느 patch가 중요한지"를 attention이 찾아낼 필요가
아예 없다.

[원본 대비 단순화]
원 논문/공식 구현(github.com/mahmoodlab/PORPOISE, models/model_utils.py::BilinearFusion)은
gating에 nn.Bilinear(두 모달리티를 함께 봐서 게이트 결정), 두 모달리티 각각 별도의 차원축소
스케일(scale_dim1/scale_dim2)을 지원한다. 여기서는 이 프로젝트의 표본 규모(TCGA train 91명)에
맞춰 파라미터 수를 최대한 억제하는 쪽으로 단순화했다 — nn.Bilinear(두 벡터의 모든 쌍을 곱하는
저용량-아닌 레이어) 대신 concat 후 일반 Linear로 게이트를 계산하고, mmhid(융합 후 차원)도
embed_dim과 같은 작은 값을 기본값으로 둔다(이 프로젝트에서 반복 확인된 교훈: 표본이 작을수록
용량을 낮춰야 한다 — `--rna-gate-only` 실험의 과적합 붕괴 참조).
"""
import torch
import torch.nn as nn


class BilinearFusion(nn.Module):
    """
    Args:
        dim1, dim2: 두 입력 모달리티(WSI, RNA)의 임베딩 차원. 이 프로젝트에서는 보통 둘 다
                    cfg.embed_dim으로 같다.
        mmhid: 융합 후 출력 차원(risk_head 입력으로 이어짐).
        gate: True면 PORPOISE와 동일하게 게이팅(각 모달리티가 서로를 보고 스스로를 얼마나
              반영할지 결정) 적용. False면 순수 outer product만(원 논문의 ablation과 동일).
        dropout: post-fusion MLP dropout.

    forward(vec1: (dim1,), vec2: (dim2,)) -> (mmhid,)
    """

    def __init__(self, dim1: int, dim2: int, mmhid: int, gate: bool = True, dropout: float = 0.25):
        super().__init__()
        self.gate = gate
        self.dim1, self.dim2 = dim1, dim2

        if gate:
            # 두 모달리티를 concat해서 보고 각자의 게이트를 계산한다(원 논문의 nn.Bilinear보다
            # 저용량) — z1/z2 모두 [vec1, vec2]를 함께 참조해 "서로를 보고 자기 자신을 얼마나
            # 반영할지" 결정한다는 원 설계 취지는 그대로 유지.
            self.gate1 = nn.Linear(dim1 + dim2, dim1)
            self.gate2 = nn.Linear(dim1 + dim2, dim2)

        # bias(상수 1)를 붙인 뒤 outer product하므로 (dim1+1)*(dim2+1)차원이 나온다 —
        # [vec1⊗vec2](진짜 pairwise 상호작용) + [vec1](bias 2 자리) + [vec2](bias 1 자리) +
        # [1](상수, bias 항)을 전부 한 벡터 안에 담는 PORPOISE의 표준 트릭.
        fusion_dim = (dim1 + 1) * (dim2 + 1)
        self.post_fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, mmhid),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        if self.gate:
            both = torch.cat([vec1, vec2], dim=-1)
            vec1 = torch.sigmoid(self.gate1(both)) * vec1
            vec2 = torch.sigmoid(self.gate2(both)) * vec2

        o1 = torch.cat([vec1, vec1.new_ones(1)], dim=-1)  # (dim1+1,)
        o2 = torch.cat([vec2, vec2.new_ones(1)], dim=-1)  # (dim2+1,)
        fusion = torch.outer(o1, o2).flatten()             # ((dim1+1)*(dim2+1),) — Kronecker product
        return self.post_fusion(fusion)                    # (mmhid,)
