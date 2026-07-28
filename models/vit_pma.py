"""
ViT_PMA — 다성분(multi-component) pooling + co-attention 기반 RNA 결합. train.py --PMA.

ViT_PM4(post-hoc sigmoid 게이트)와 달리, RNA가 4개 pooling 관점(mean/std/attention-weighted/
top-k-mean) 중 "이 환자의 RNA subtype에는 어떤 관점이 더 중요한가"를 co-attention으로 직접
골라 가중합한다 — ViT_M4A(패치 N개 전체에 대한 co-attention)의 아이디어를, 훨씬 작고
해석 가능한 "4개 통계적 관점의 집합"에 적용한 버전. CoAttentionPooling(vit_m4a.py)은
key/value 개수에 무관하게 동작해(patch N개든 component 4개든) 그대로 재사용한다.
"""
import torch
import torch.nn as nn

from .vit_m1 import ViT_M1
from .vit_m4a import CoAttentionPooling
from .spatial_features import spatial_autocorr, attention_dispersion
from .multi_component_pooling import MultiComponentPooling
from .clinical_encoder import ClinicalEncoder
from .rna_encoder import RNAEncoder
from config import ModelConfig

# 2026-07-21: 레퍼런스 인코더 폭 비율(RNA=256, Clinical=16, WSI엔 안 맞춤)을 RNA 전처리
# 버그 수정 이후 PMA_EX_SS_AUX로 검증했으나 negative(external C 0.611->0.601, tile-fusion
# 단독보다도 하락 — findings_backlog.md 최상위 발견 항목) — 원래의 균일 cfg.embed_dim
# 형태로 되돌림. CoAttentionPooling의 context_dim 파라미터(vit_m4a.py)는 인프라로 남겨둔다.
#
# 2026-07-21(2차): 위 실험은 WSI 차원(cfg.embed_dim=64)까지 통째로 키운 것이 원인일 수
# 있다는 가설로, WSI는 64로 고정하고 RNA/Clinical만 레퍼런스 비율 감각(RNA>WSI>Clinical)에
# 맞춰 절대 크기를 축소한 RNA=128/Clinical=16 조합을 rna_dim/clinical_dim으로 재시도한다
# (train.py --rna-dim/--clinical-dim, --PMA 전용).
#
# 2026-07-21(3차): scripts/diagnose_wsi_reliance.py·diagnose_wsi_gradients.py 진단 결과
# (findings_backlog.md 최상위 발견 2차) — WSI ablation(z_wsi=0/셔플)이 internal/external
# 성능에 거의 영향이 없고, RNA 인코더 gradient norm이 학습 내내 WSI 브랜치의 ~4배. risk_head가
# z_rna(co-attention을 거치지 않은 원본 RNA 임베딩)를 직결 concat으로 그냥 받을 수 있다는 게
# WSI 브랜치를 우회하는 "지름길"일 수 있다는 가설로, rna_gate_only=True면 z_rna를
# component_coattn의 query(WSI 4관점 중 고르는 용도)로만 쓰고 risk_head에는 [z_wsi, z_clinical]만
# 넣는다 — RNA 정보는 여전히 z_wsi에 "녹아들어" 있지만 우회 경로는 차단된다.


class ViT_PMA(ViT_M1):
    def __init__(
        self,
        cfg: ModelConfig,
        age_mean: float,
        age_std: float,
        rna_input_dim: int,
        precomputed: bool = True,
        backbone: str = "resnet50",
        num_heads: int = 2,
        use_staging: bool = False,
        stage_stats: dict[str, tuple[float, float]] | None = None,
        rna_dim: int | None = None,
        clinical_dim: int | None = None,
        rna_gate_only: bool = False,
    ):
        super().__init__(cfg, precomputed, backbone)
        rna_dim = rna_dim or cfg.embed_dim
        clinical_dim = clinical_dim or cfg.embed_dim
        self.rna_gate_only = rna_gate_only
        self.attn_pool = MultiComponentPooling(cfg.embed_dim)
        self.component_coattn = CoAttentionPooling(
            cfg.embed_dim, num_heads=num_heads, dropout=cfg.dropout, context_dim=rna_dim
        )

        self.clinical_encoder = ClinicalEncoder(
            clinical_dim, age_mean, age_std, use_staging=use_staging, stage_stats=stage_stats
        )
        self.rna_encoder = RNAEncoder(rna_input_dim, rna_dim, dropout=cfg.dropout)
        # 2026-07-23: 학습형 spatial attention(kNN/hybrid) 전부가 pre-augment에서 baseline을
        # 못 넘은 뒤(findings_backlog.md), "새 attention 파라미터 자체가 과적합 유인"이라는
        # 가설을 검증하기 위한 대안 — 좌표/패치임베딩/attn_weights에서 결정론적으로 계산되는
        # 스칼라(models/spatial_features.py, 학습 파라미터 없음)만 risk_head 5번째 관점으로
        # 추가한다. 둘 다 독립적으로 켤 수 있다(순차 검증용).
        self.use_spatial_autocorr = getattr(cfg, "use_spatial_autocorr", False)
        self.use_attn_dispersion = getattr(cfg, "use_attn_dispersion", False)
        spatial_feat_dim = (2 if self.use_spatial_autocorr else 0) + (1 if self.use_attn_dispersion else 0)
        # risk_head 입력: [z_wsi(WSI_D), z_clinical(clinical_dim)] (+ z_rna(rna_dim), rna_gate_only=False일 때만)
        # (+ spatial_feat(spatial_feat_dim), 위 두 플래그 중 하나라도 켜졌을 때만)
        # (rna_dim/clinical_dim이 둘 다 기본값(None)이고 rna_gate_only=False, spatial_feat_dim=0이면
        # 3*cfg.embed_dim과 동일 — 기존 동작 보존)
        risk_input_dim = cfg.embed_dim + clinical_dim + (0 if rna_gate_only else rna_dim) + spatial_feat_dim
        self.risk_head = nn.Sequential(
            nn.LayerNorm(risk_input_dim),
            nn.Linear(risk_input_dim, 1),
        )

    def forward(
        self,
        coords: torch.Tensor,
        patch_paths=None,
        features: torch.Tensor | None = None,
        transform=None,
        chunk_size: int | None = None,
        rna_context: torch.Tensor | None = None,  # 사용 안 함(train.py 호출 시그니처 호환용)
        tile_cache: dict | None = None,
    ) -> dict:
        patch_tokens = self._patch_tokens(coords, patch_paths, features, transform, chunk_size, tile_cache)
        ctx_tokens = self.vit(patch_tokens, coords)
        components, attn_weights = self.attn_pool(ctx_tokens)  # (4, D), (N,)
        # meanpool_embed: --rna-aux-weight(models/rna_predictor.py) 보조과제 입력 전용.
        # patch_tokens: scripts/train_spatial_residual.py(공간정보 잔차 branch) 전용 — Nystrom
        # *이전* raw CNN 출력을 그대로 노출해, 이미 근사/혼합된 ctx_tokens가 아니라 패치별
        # 독립적인 표현 위에서 kNN 그래프를 새로 구성할 수 있게 한다.
        out = {
            "embed": components, "attn_weights": attn_weights,
            "meanpool_embed": ctx_tokens.mean(dim=0), "patch_tokens": patch_tokens,
        }
        # 학습 파라미터 없는 공간 특징(models/spatial_features.py) — --spatial-autocorr/
        # --attn-dispersion으로 독립적으로 켠다.
        if self.use_spatial_autocorr or self.use_attn_dispersion:
            feats = []
            if self.use_spatial_autocorr:
                feats.append(spatial_autocorr(patch_tokens, coords))
            if self.use_attn_dispersion:
                feats.append(attention_dispersion(coords, attn_weights))
            out["spatial_feat"] = torch.cat(feats, dim=0)  # (spatial_feat_dim,)
        return out

    def encode_rna(self, rna: torch.Tensor) -> torch.Tensor:
        return self.rna_encoder(rna.unsqueeze(0)).squeeze(0)

    def combine_with_clinical_rna(
        self,
        patient_embed: torch.Tensor,  # (4, D) — 환자 단위로 평균 풀링된 4개 관점
        age_years: torch.Tensor,
        sex_idx: torch.Tensor,
        z_rna: torch.Tensor,
        stage_ord: dict[str, torch.Tensor] | None = None,  # self.clinical_encoder.use_staging=True일 때만 필요
        spatial_feat: torch.Tensor | None = None,  # (spatial_feat_dim,) — 환자 단위 평균, models/spatial_features.py
    ) -> torch.Tensor:
        stage_kwargs = {}
        if stage_ord is not None:
            stage_kwargs["stage_ord"] = {k: v.unsqueeze(0) for k, v in stage_ord.items()}
        z_clinical = self.clinical_encoder(
            age_years.unsqueeze(0), sex_idx.unsqueeze(0), **stage_kwargs
        ).squeeze(0)  # (D,)
        z_wsi, _ = self.component_coattn(patient_embed, z_rna)  # (D,) — RNA가 4개 관점 중 골라 가중합
        if self.rna_gate_only:
            # z_rna는 위 co-attention의 query로만 관여하고, risk_head에는 직결 concat하지 않는다 —
            # RNA로 곧장 우회하는 "지름길"을 막아 risk_head가 z_wsi(에 이미 녹아든 RNA 정보)와
            # z_clinical만으로 예측하도록 강제한다.
            fused = torch.cat([z_wsi, z_clinical], dim=-1)       # (2D,)
        else:
            fused = torch.cat([z_wsi, z_clinical, z_rna], dim=-1)  # (3D,)
        if spatial_feat is not None:
            fused = torch.cat([fused, spatial_feat], dim=-1)
        return fused
