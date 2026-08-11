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
from .clinical_encoder import ClinicalEncoder, STAGE_FIELDS, _STAGE_BUFFER_NAMES
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
        use_clinical: bool = True,
        use_margin: bool = False,
        margin_stats: tuple[float, float] | None = None,
        use_age_sex: bool = True,
        combine_mode: str = "concat",
        drop_component: str | None = None,
        top_frac: float = 0.1,
    ):
        super().__init__(cfg, precomputed, backbone)
        if combine_mode not in ("concat", "cox_add"):
            raise ValueError(f"알 수 없는 combine_mode: {combine_mode}")
        rna_dim = rna_dim or cfg.embed_dim
        clinical_dim = clinical_dim or cfg.embed_dim
        self.rna_gate_only = rna_gate_only
        # 2026-07-28: M3(WSI+RNA, clinical 제외) ablation용 — PMA_EX_SS_AUX(M4 슬롯) 구조 그대로
        # 두고 clinical concat만 뺀다(train.py --no-clinical). False면 clinical_encoder 자체를
        # 안 만들고 combine_with_clinical_rna()도 z_clinical을 계산/concat하지 않는다.
        self.use_clinical = use_clinical
        # 2026-08-05: train_light.py --M7 --combine-mode cox_add를 PMA에 이식(train.py
        # --combine-mode). "concat"(기본)은 clinical_encoder(MLP)→z_clinical을 [z_wsi, z_rna]에
        # concat, "cox_add"는 clinical을 임베딩하지 않고 risk_head(z_wsi+z_rna) 스칼라에 고전적
        # Cox 가산항(clinical_linear, zero-init)으로 직접 더한다 — M7에서 cox_add가 R_ONLY(margin
        # 단독)의 internal을 M6 수준까지 끌어올린 효과가 PMA에도 재현되는지 검증.
        self.combine_mode = combine_mode
        self.use_margin = use_margin
        self.use_age_sex = use_age_sex
        self.use_staging = use_staging
        # 2026-08-09: scripts/diagnose_pma_component_reliance.py의 zero-ablation 진단(4개 성분
        # 다 지워봐도 손해가 없었음)에 이어, 구조적으로 하나를 빼고 재학습하는 ablation용
        # (train.py --drop-component). 기본 None이면 기존과 완전히 동일(4개 다 사용).
        # top_frac(train.py --top-frac)도 같이 노출 — top-k 성분이 상위 10%라는 너무 작은
        # 부분집합이라 노이즈에 민감했을 수 있다는 가설(top-k를 지우면 internal이 올랐던 것과
        # 같은 방향)을 검증한다. top_frac을 키우면(예: 0.25) 같은 "attention 상위" 컨셉은
        # 유지하되 표본을 넓혀 노이즈를 줄일 수 있는지 본다.
        self.attn_pool = MultiComponentPooling(cfg.embed_dim, exclude=drop_component, top_frac=top_frac)
        self.component_coattn = CoAttentionPooling(
            cfg.embed_dim, num_heads=num_heads, dropout=cfg.dropout, context_dim=rna_dim
        )

        if combine_mode == "concat":
            if self.use_clinical:
                self.clinical_encoder = ClinicalEncoder(
                    clinical_dim, age_mean, age_std, use_staging=use_staging, stage_stats=stage_stats,
                    use_margin=use_margin, margin_stats=margin_stats, use_age_sex=use_age_sex,
                )
        else:  # cox_add
            if not self.use_clinical:
                raise ValueError("combine_mode='cox_add'는 use_clinical=True에서만 의미가 있습니다.")
            self.register_buffer("age_mean", torch.tensor(age_mean, dtype=torch.float32))
            self.register_buffer("age_std", torch.tensor(age_std, dtype=torch.float32))
            if use_margin:
                m_mean, m_std = margin_stats
                self.register_buffer("margin_mean", torch.tensor(m_mean, dtype=torch.float32))
                self.register_buffer("margin_std", torch.tensor(m_std, dtype=torch.float32))
            if use_staging:
                # 2026-08-06: cox_add에 staging(T/N/M/grade) 추가 — ClinicalEncoder(concat 모드)와
                # 동일하게 필드당 (z_score, known_flag) 2차원씩, 총 4필드=8차원을 raw_dim에 더한다.
                if stage_stats is None:
                    raise ValueError("use_staging=True면 stage_stats가 필요합니다.")
                for field in STAGE_FIELDS:
                    mean, std = stage_stats[field]
                    short = _STAGE_BUFFER_NAMES[field]
                    self.register_buffer(f"{short}_mean", torch.tensor(mean, dtype=torch.float32))
                    self.register_buffer(f"{short}_std", torch.tensor(std, dtype=torch.float32))
            raw_dim = (2 if use_age_sex else 0) + (2 if use_margin else 0) + (2 * len(STAGE_FIELDS) if use_staging else 0)
            if raw_dim == 0:
                raise ValueError("use_age_sex=False이고 use_margin=False이고 use_staging=False면 clinical 입력이 없습니다.")
            self.clinical_linear = nn.Linear(raw_dim, 1, bias=False)
            nn.init.zeros_(self.clinical_linear.weight)  # 초기엔 risk_head(z_wsi+z_rna)와 동일
        self.rna_encoder = RNAEncoder(rna_input_dim, rna_dim, dropout=cfg.dropout)
        # 2026-07-23: 학습형 spatial attention(kNN/hybrid) 전부가 pre-augment에서 baseline을
        # 못 넘은 뒤(findings_backlog.md), "새 attention 파라미터 자체가 과적합 유인"이라는
        # 가설을 검증하기 위한 대안 — 좌표/패치임베딩/attn_weights에서 결정론적으로 계산되는
        # 스칼라(models/spatial_features.py, 학습 파라미터 없음)만 risk_head 5번째 관점으로
        # 추가한다. 둘 다 독립적으로 켤 수 있다(순차 검증용).
        self.use_spatial_autocorr = getattr(cfg, "use_spatial_autocorr", False)
        self.use_attn_dispersion = getattr(cfg, "use_attn_dispersion", False)
        if self.use_attn_dispersion:
            # 2026-07-30: vit_m1.py와 동일한 이유(dispersion 원값 스케일이 나머지 risk_head
            # 입력보다 5~10배 커서 LayerNorm 통계를 왜곡할 수 있음) — 학습되는 배율로 낮춘다.
            self.dispersion_scale = nn.Parameter(torch.tensor(0.2))
        spatial_feat_dim = (2 if self.use_spatial_autocorr else 0) + (1 if self.use_attn_dispersion else 0)
        # risk_head 입력: [z_wsi(WSI_D)] (+ z_clinical(clinical_dim), use_clinical=True이고
        # combine_mode="concat"일 때만 — cox_add는 clinical을 risk_head에 넣지 않고 최종
        # 스칼라에 별도로 더한다) (+ z_rna(rna_dim), rna_gate_only=False일 때만)
        # (+ spatial_feat(spatial_feat_dim), 위 두 플래그 중 하나라도 켜졌을 때만)
        # (rna_dim/clinical_dim이 둘 다 기본값(None)이고 use_clinical=True, combine_mode="concat",
        # rna_gate_only=False, spatial_feat_dim=0이면 3*cfg.embed_dim과 동일 — 기존 동작 보존)
        risk_input_dim = (
            cfg.embed_dim
            + (clinical_dim if (self.use_clinical and combine_mode == "concat") else 0)
            + (0 if rna_gate_only else rna_dim)
            + spatial_feat_dim
        )
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
                feats.append(attention_dispersion(coords, attn_weights) * self.dispersion_scale)
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
        margin_ord: torch.Tensor | None = None,  # self.clinical_encoder.use_margin=True일 때만 필요
        spatial_feat: torch.Tensor | None = None,  # (spatial_feat_dim,) — 환자 단위 평균, models/spatial_features.py
    ) -> torch.Tensor:
        clinical_kwargs = {}
        if stage_ord is not None:
            clinical_kwargs["stage_ord"] = {k: v.unsqueeze(0) for k, v in stage_ord.items()}
        if margin_ord is not None:
            clinical_kwargs["margin_ord"] = margin_ord.unsqueeze(0)
        z_wsi, _ = self.component_coattn(patient_embed, z_rna)  # (D,) — RNA가 4개 관점 중 골라 가중합
        parts = [z_wsi]
        if self.use_clinical and self.combine_mode == "concat":
            # combine_mode="cox_add"면 clinical은 여기서 임베딩/concat되지 않고, 호출부(train.py)가
            # risk_head(이 fused 임베딩) 계산 뒤 self.clinical_linear(raw)를 별도로 더한다.
            z_clinical = self.clinical_encoder(
                age_years.unsqueeze(0), sex_idx.unsqueeze(0), **clinical_kwargs
            ).squeeze(0)  # (D,)
            parts.append(z_clinical)
        if not self.rna_gate_only:
            # rna_gate_only=True면 z_rna는 위 co-attention의 query로만 관여하고, risk_head에는
            # 직결 concat하지 않는다 — RNA로 곧장 우회하는 "지름길"을 막아 risk_head가
            # z_wsi(에 이미 녹아든 RNA 정보)와 z_clinical만으로 예측하도록 강제한다.
            parts.append(z_rna)
        fused = torch.cat(parts, dim=-1)
        if spatial_feat is not None:
            fused = torch.cat([fused, spatial_feat], dim=-1)
        return fused

    def _clinical_raw(self, age_years: torch.Tensor, sex_idx: torch.Tensor,
                       margin_ord: torch.Tensor | None = None,
                       stage_ord: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """combine_mode="cox_add" 전용 — models/clinical_rna_only.py::ClinicalRNAOnly._clinical_raw와 동일 관례.
        stage_ord: self.use_staging=True일 때만 필요. {field: () 스칼라 long} — encode_stage_value() 규약."""
        feats = []
        if self.use_age_sex:
            age_z = (age_years.float() - self.age_mean) / self.age_std
            feats += [age_z, sex_idx.float()]
        if self.use_margin:
            ordv = margin_ord.float()
            known = (ordv >= 0).float()
            z = torch.where(ordv >= 0, (ordv - self.margin_mean) / self.margin_std, torch.zeros_like(ordv))
            feats += [z, known]
        if self.use_staging:
            for field in STAGE_FIELDS:
                short = _STAGE_BUFFER_NAMES[field]
                ordv = stage_ord[field].float()
                known = (ordv >= 0).float()
                mean = getattr(self, f"{short}_mean")
                std = getattr(self, f"{short}_std")
                z = torch.where(ordv >= 0, (ordv - mean) / std, torch.zeros_like(ordv))
                feats += [z, known]
        return torch.stack(feats, dim=-1).unsqueeze(0)  # (1, raw_dim)
