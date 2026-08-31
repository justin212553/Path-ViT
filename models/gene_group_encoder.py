"""
GeneGroupEncoder — RNA z-score 벡터를 data/select_rnaseq_genes.py::PDAC_LITERATURE_GENE_SETS
(8개 PDAC 기능별 유전자 카테고리)로 나눠, 카테고리마다 독립된 학습되는 선형층으로
(embed_dim,) 토큰 하나씩을 만든다 — MCAT(Chen et al. 2021)/SurvPath(Jaume et al. 2024)의
"pathway token" 방식.

2026-08-31: findings_backlog.md "10. Pathway8 집계 — 실패"(--rna-genes pathway8, external
C 0.49~0.52로 동전 던지기 이하)의 원인 수정판이다. 실패 원인은 "카테고리로 나눈 것"이 아니라
"카테고리 안에서 유전자별 z-score를 방향성 없이(unsigned) 그냥 평균낸 것" — 예를 들어
immune_inflammation 카테고리 안에 CD8A(좋은 예후)와 FOXP3(나쁜 예후)가 같이 있으면 단순
평균은 서로 상쇄돼 신호가 사라진다. 여기서는 카테고리별로 "평균"이 아니라 "학습되는 선형결합"
(nn.Linear, 유전자마다 부호/크기가 다른 가중치를 학습)을 써서 이 상쇄 문제를 구조적으로
피한다 — 데이터 단계에서 미리 뭉개지 않고, 원본 유전자별 z-score를 그대로 모델에 넣는다.
"""
from pathlib import Path

import torch
import torch.nn as nn


class GeneGroupEncoder(nn.Module):
    """
    Args:
        gene_ids: 학습에 쓰는 전체 RNA feature 순서(예: literature_guided_gene_ids_intersection
                  결과) — RNA 입력 텐서 (G,)의 각 위치가 어떤 gene_id인지 알려준다.
        gene_sets: {category_name: [gene_id, ...]} — data/dataset.py::pathway_category_gene_ids()
                   (또는 동일 형식의 다른 딕셔너리)로 얻은 카테고리별 유전자 목록.
        embed_dim: 카테고리 토큰 1개의 출력 차원(WSI patch 토큰과 같은 폭이어야 co-attention에
                   그대로 쓸 수 있음).

    forward(rna: (G,)) -> (K, embed_dim) — K = gene_ids와 실제로 겹치는 카테고리 수(보통 8,
    이 RNA feature 목록에 해당 카테고리 유전자가 하나도 없으면 그 카테고리는 자동 제외됨).
    """

    def __init__(self, gene_ids: list[str], gene_sets: dict[str, list[str]], embed_dim: int):
        super().__init__()
        gene_pos = {g: i for i, g in enumerate(gene_ids)}
        self.categories: list[str] = []
        self.encoders = nn.ModuleDict()
        for cat, genes in gene_sets.items():
            idx = sorted(gene_pos[g] for g in genes if g in gene_pos)
            if len(idx) == 0:
                continue  # 이 gene_ids 목록엔 이 카테고리 유전자가 하나도 없음 — 조용히 skip
            self.register_buffer(f"_idx_{cat}", torch.tensor(idx, dtype=torch.long))
            # 카테고리마다 독립된 raw Linear(bias 없음) — GELU/hidden layer 없이 최대한 단순하게
            # (이 프로젝트에서 반복 확인된 교훈: 신호가 약한/작은 입력엔 MLP보다 단순한 선형결합이
            # 낫다, models/clinical_rna_only.py의 clinical cox_add ablation 참조). LayerNorm은
            # 카테고리별 유전자 수가 13~30개로 들쭉날쭉해 스케일을 맞추는 용도로 유지.
            self.encoders[cat] = nn.Sequential(
                nn.LayerNorm(len(idx)),
                nn.Linear(len(idx), embed_dim, bias=False),
            )
            self.categories.append(cat)
        if len(self.categories) == 0:
            raise ValueError("gene_ids와 겹치는 gene_sets 카테고리가 하나도 없습니다.")

    @property
    def num_groups(self) -> int:
        return len(self.categories)

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        """rna: (G,) — 전체 z-score 벡터. Returns: (K, embed_dim)."""
        tokens = []
        for cat in self.categories:
            idx = getattr(self, f"_idx_{cat}")
            sub = rna.index_select(0, idx)  # (len(idx),)
            tokens.append(self.encoders[cat](sub.unsqueeze(0)).squeeze(0))
        return torch.stack(tokens, dim=0)  # (K, D)
