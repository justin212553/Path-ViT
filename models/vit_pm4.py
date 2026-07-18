"""
ViT_PM4 — 다성분(multi-component) pooling + post-hoc RNA 게이트. train.py --PM4.

findings_backlog.md 1번 항목(ABMIL 단일 벡터 압축 → 다성분 pooling)의 첫 구현. 레퍼런스
M3/M4의 "Morphology Burden Pooling + RNA-guided gating" 설계를 그대로 이식한 버전 —
pooling 자체는 RNA와 무관하게(순수 WSI 관점 4개) 먼저 만들고, RNA는 그 결과를 post-hoc으로
게이팅한다. "원본 WSI 표현"과 "RNA로 조절된 표현"을 [H_i, H_i_gated, z_clinical, z_rna]로
모두 risk_head에 넘겨, 어느 쪽을 얼마나 신뢰할지는 risk_head가 학습하게 한다(레퍼런스가
"원본 WSI 정보를 보존하면서"라고 명시한 설계 의도와 동일).

ViT_M4B(patch token 자체에 FiLM)와 달리, 여기서는 pooling 시점까지 RNA가 전혀 개입하지
않는다 — MultiComponentPooling이 만드는 4개 관점이 RNA로 미리 물들지 않은 순수 WSI 요약을
유지해야 게이트가 "무엇을 조절하는지"가 명확해지기 때문이다(M4B가 다성분 pooling과 궁합이
안 좋은 이유이기도 하다).
"""
import torch
import torch.nn as nn

from .vit_m1 import ViT_M1
from .multi_component_pooling import MultiComponentPooling
from .clinical_encoder import ClinicalEncoder
from .rna_encoder import RNAEncoder
from config import ModelConfig


class ViT_PM4(ViT_M1):
    def __init__(
        self,
        cfg: ModelConfig,
        age_mean: float,
        age_std: float,
        rna_input_dim: int,
        precomputed: bool = True,
        backbone: str = "resnet50",
        use_staging: bool = False,
        stage_stats: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__(cfg, precomputed, backbone)
        self.attn_pool = MultiComponentPooling(cfg.embed_dim)
        pooled_dim = MultiComponentPooling.NUM_COMPONENTS * cfg.embed_dim  # H_i flatten 차원 (4D)

        self.clinical_encoder = ClinicalEncoder(
            cfg.embed_dim, age_mean, age_std, use_staging=use_staging, stage_stats=stage_stats
        )
        self.rna_encoder = RNAEncoder(rna_input_dim, cfg.embed_dim, dropout=cfg.dropout)
        self.rna_gate = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, pooled_dim),
            nn.Sigmoid(),
        )
        # risk_head 입력: [H_i(4D), H_i_gated(4D), z_clinical(D), z_rna(D)] = 10D
        self.risk_head = nn.Sequential(
            nn.LayerNorm(pooled_dim * 2 + cfg.embed_dim * 2),
            nn.Linear(pooled_dim * 2 + cfg.embed_dim * 2, 1),
        )

    def forward(
        self,
        coords: torch.Tensor,
        patch_paths=None,
        features: torch.Tensor | None = None,
        transform=None,
        chunk_size: int | None = None,
        rna_context: torch.Tensor | None = None,  # 사용 안 함(train.py 호출 시그니처 호환용) — pooling은 RNA와 무관
    ) -> dict:
        patch_tokens = self._patch_tokens(coords, patch_paths, features, transform, chunk_size)
        ctx_tokens = self.vit(patch_tokens, coords)
        components, attn_weights = self.attn_pool(ctx_tokens)  # (4, D), (N,)
        h_i = components.flatten()  # (4D,)
        return {"embed": h_i, "attn_weights": attn_weights}

    def encode_rna(self, rna: torch.Tensor) -> torch.Tensor:
        return self.rna_encoder(rna.unsqueeze(0)).squeeze(0)

    def combine_with_clinical_rna(
        self,
        patient_embed: torch.Tensor,  # (4D,) — 환자 단위로 평균 풀링된 H_i
        age_years: torch.Tensor,
        sex_idx: torch.Tensor,
        z_rna: torch.Tensor,
        stage_ord: dict[str, torch.Tensor] | None = None,  # self.clinical_encoder.use_staging=True일 때만 필요
    ) -> torch.Tensor:
        stage_kwargs = {}
        if stage_ord is not None:
            stage_kwargs["stage_ord"] = {k: v.unsqueeze(0) for k, v in stage_ord.items()}
        z_clinical = self.clinical_encoder(
            age_years.unsqueeze(0), sex_idx.unsqueeze(0), **stage_kwargs
        ).squeeze(0)  # (D,)
        gate = self.rna_gate(z_rna)               # (4D,)
        h_i_gated = patient_embed * gate           # (4D,)
        return torch.cat([patient_embed, h_i_gated, z_clinical, z_rna], dim=-1)  # (10D,)
