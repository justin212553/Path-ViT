"""
ViT_MCAT — 진짜 MCAT(Chen et al. 2021)/SurvPath(Jaume et al. 2024) 스타일 multi-pathway-token
co-attention. train.py --MCAT.

2026-08-31 배경: models/vit_m4a.py::ViT_M4A(--M4A)가 이미 "genomic query가 patch에 직접
co-attention" 아이디어를 이식했지만, RNA 전체를 RNAEncoder로 압축한 **단일** (D,) 벡터
하나만 query로 썼다 — docstring에 스스로 "유전자를 pathway별로 안 쪼갠 단순화 버전"이라고
명시돼 있다. 반면 실제 MCAT/SurvPath는 유전자를 6~50개의 pathway 토큰으로 나눠 **여러 개의
query**가 병렬로 patch 전체를 본다 — query 1개 vs key 몇 개(우리 PMA는 4개 pooling 관점)
같은 저용량 구조에서는 co-attention이 균등분포로 수렴하는 게 오히려 자연스러운 결과에
가깝다(findings_backlog.md 최상위 발견, attention entropy 0.999~1.000 붕괴 참조) — 이
모델은 query 개수 자체를 늘려 그 병목을 정면으로 없앤다.

ViT_M4A와 마찬가지로 ViT_M4를 상속하고 attn_pool/encode_rna/combine_with_clinical_rna
세 곳만 교체한다(fusion 골격 자체는 동일: encode_rna() → 슬라이드별 attn_pool(patch,
rna_context) → 환자 단위 평균 풀링 → combine_with_clinical_rna() → risk_head).

[RNA 표현 두 갈래]
  - attn_pool(co-attention)의 query: GeneGroupEncoder가 만든 (K, D) pathway 토큰 그대로
    (K=8, data/select_rnaseq_genes.py::PDAC_LITERATURE_GENE_SETS) — encode_rna()가 이걸 반환.
  - combine_with_clinical_rna()의 risk_head 직결 concat: (K,D) 토큰을 K축으로 평균한
    (D,) 요약 벡터 하나(ViT_M4의 [z_wsi, z_clinical, z_rna] concat 계약과 차원을 맞추기
    위함) — 병렬 attention은 K개 다 쓰지만, 최종 concat까지 K배로 불릴 필요는 없다는 판단.

self.rna_encoder(부모 ViT_M4가 만드는 단일-벡터 RNAEncoder)는 이 모델에서 실제로는 안
쓰이지만 **일부러 지우지 않는다** — train.py::_patient_risk가 `hasattr(model, "rna_encoder")`
로 "이 모델이 RNA를 쓰는가"를 판단하기 때문에(encode_rna 메서드 존재 여부가 아니라 속성
존재 여부로 분기), 지우면 RNA 인코딩 자체가 호출되지 않는 조용한 버그가 된다.
"""
import torch
import torch.nn as nn

from .vit_m4 import ViT_M4
from .gene_group_encoder import GeneGroupEncoder
from config import ModelConfig


class MultiQueryCoAttentionPooling(nn.Module):
    """models/vit_m4a.py::CoAttentionPooling의 다중 쿼리 버전 — RNA를 단일 (D,) 벡터가 아니라
    K개 pathway 토큰(K,D)으로 받아 전부 동시에 patch 전체에 cross-attention한다(진짜 MCAT처럼
    여러 genomic query가 병렬로 patch를 본다). K개 attended 결과는 평균해서 하나의 z_wsi(D,)로
    합치고, attention 가중치도 K개 평균으로 (N,) 하나만 반환한다 — models/vit_m1.py::
    AttentionPooling과 반환 규약을 그대로 맞춰 ViT_M1.forward()/attention_dispersion 등
    기존 코드를 전혀 안 건드리고 drop-in으로 쓸 수 있게 한다.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens:  (N, D) — ViT를 지난 patch 토큰(key/value)
            context: (K, D) — GeneGroupEncoder가 만든 pathway 토큰들(query)
        Returns:
            wsi_embed:    (D,) — K개 query 결과 평균
            attn_weights: (N,) — K개 query의 patch attention을 평균한 단일 분포(시각화/
                          attention_dispersion용, ViT_M1::AttentionPooling과 동일 규약)
        """
        query = context.unsqueeze(0)   # (1, K, D)
        kv = tokens.unsqueeze(0)       # (1, N, D)
        attn_out, attn_weights = self.mha(
            query, kv, kv, need_weights=True, average_attn_weights=True
        )  # attn_out: (1, K, D), attn_weights: (1, K, N)
        wsi_embed = attn_out.squeeze(0).mean(dim=0)              # (D,)
        attn_weights = attn_weights.squeeze(0).mean(dim=0)       # (N,) — K개 query 평균
        return wsi_embed, attn_weights


class ViT_MCAT(ViT_M4):
    def __init__(
        self,
        cfg: ModelConfig,
        age_mean: float,
        age_std: float,
        rna_input_dim: int,
        gene_ids: list[str],
        gene_sets: dict[str, list[str]] | None = None,
        precomputed: bool = True,
        backbone: str = "resnet50",
        num_heads: int = 4,
        use_staging: bool = False,
        stage_stats: dict[str, tuple[float, float]] | None = None,
        use_margin: bool = False,
        margin_stats: tuple[float, float] | None = None,
        use_age_sex: bool = True,
        combine_mode: str = "concat",
        use_attn_dispersion: bool = False,
        skip_patch_vit: bool = False,
    ):
        super().__init__(cfg, age_mean, age_std, rna_input_dim, precomputed, backbone,
                          use_staging=use_staging, stage_stats=stage_stats,
                          use_margin=use_margin, margin_stats=margin_stats, use_age_sex=use_age_sex,
                          combine_mode=combine_mode, use_attn_dispersion=use_attn_dispersion,
                          skip_patch_vit=skip_patch_vit)
        if gene_sets is None:
            from data.select_rnaseq_genes import PDAC_LITERATURE_GENE_SETS
            gene_sets = PDAC_LITERATURE_GENE_SETS
        self.gene_group_encoder = GeneGroupEncoder(gene_ids, gene_sets, cfg.embed_dim)
        self.attn_pool = MultiQueryCoAttentionPooling(cfg.embed_dim, num_heads=num_heads, dropout=cfg.dropout)

    def encode_rna(self, rna: torch.Tensor) -> torch.Tensor:
        """rna: (G,) 전체 z-score 벡터. Returns: (K, D) — GeneGroupEncoder의 pathway 토큰들
        (ViT_M4.encode_rna의 (D,) 계약과 다름 — attn_pool이 K개 query를 그대로 받아써야 하므로
        여기서 미리 평균내지 않는다. risk_head 쪽 요약은 combine_with_clinical_rna에서 처리)."""
        return self.gene_group_encoder(rna)

    def combine_with_clinical_rna(
        self,
        patient_embed: torch.Tensor,
        age_years: torch.Tensor,
        sex_idx: torch.Tensor,
        z_rna: torch.Tensor,  # (K, D) — encode_rna() 출력 그대로
        stage_ord: dict[str, torch.Tensor] | None = None,
        margin_ord: torch.Tensor | None = None,
        spatial_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ViT_M4.combine_with_clinical_rna는 z_rna가 (D,)라고 가정하고 그대로 concat한다 —
        여기서는 K개 pathway 토큰을 K축으로 평균한 (D,) 요약 하나로 줄인 뒤 부모 구현에
        위임한다(risk_head 입력 차원 계약을 그대로 재사용하기 위함, 코드 중복 방지)."""
        z_rna_summary = z_rna.mean(dim=0)  # (K, D) -> (D,)
        return super().combine_with_clinical_rna(
            patient_embed, age_years, sex_idx, z_rna_summary,
            stage_ord=stage_ord, margin_ord=margin_ord, spatial_feat=spatial_feat,
        )
