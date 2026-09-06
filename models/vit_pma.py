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
from .clinical_encoder import (
    ClinicalEncoder, STAGE_FIELDS, _STAGE_BUFFER_NAMES, MUTATION_FIELDS, _MUTATION_BUFFER_NAMES,
)
from .rna_encoder import RNAEncoder
from .tumor_content_head import TumorContentHead
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
        use_mutation: bool = False,
        mutation_stats: dict[str, tuple[float, float]] | None = None,
        use_age_sex: bool = True,
        combine_mode: str = "concat",
        drop_component: str | None = None,
        top_frac: float = 0.1,
        rna_combine_mode: str = "concat",
        skip_patch_vit: bool = False,
        use_tumor_type_embed: bool = False,
        use_tile_risk_head: bool = False,
        use_coord_embed: bool = False,
        coord_embed_concat: bool = False,
        coord_embed_learnable_scale: bool = False,
        coord_embed_shuffle: bool = False,
        use_wsi_extra_mlp: bool = False,
        use_coattn: bool = True,
        surv_n_classes: int = 1,
        cluster_pool: bool = False,
        cluster_centroids_path: str | None = None,
        cluster_pool_after_vit: bool = False,
        cluster_pool_temperature: float | None = None,
        tumor_content_head_path: str | None = None,
    ):
        super().__init__(cfg, precomputed, backbone, skip_patch_vit=skip_patch_vit,
                          use_tumor_type_embed=use_tumor_type_embed,
                          use_coord_embed=use_coord_embed, coord_embed_concat=coord_embed_concat,
                          coord_embed_learnable_scale=coord_embed_learnable_scale,
                          coord_embed_shuffle=coord_embed_shuffle,
                          use_wsi_extra_mlp=use_wsi_extra_mlp)
        if combine_mode not in ("concat", "cox_add"):
            raise ValueError(f"알 수 없는 combine_mode: {combine_mode}")
        if rna_combine_mode not in ("concat", "cox_add"):
            raise ValueError(f"알 수 없는 rna_combine_mode: {rna_combine_mode}")
        if use_mutation and combine_mode != "cox_add":
            # models/vit_m4.py::ViT_M4와 달리 ViT_PMA는 combine_mode="concat"일 때 clinical을
            # ClinicalEncoder(MLP)에 통째로 맡기고(mutation 미지원) cox_add일 때만 raw feature를
            # 직접 다룬다(_clinical_embed) — mutation은 이 raw feature 경로에만 이식했다.
            raise ValueError("use_mutation=True는 combine_mode='cox_add'에서만 지원합니다.")
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
        # 2026-08-09: clinical의 cox_add 원리를 RNA에도 이식 — RNA는 여전히 component_coattn의
        # query로 WSI pooling을 guide하지만(M1_POOL 대비 M3에서 이미 유효성이 확인된 경로,
        # 이건 그대로 둔다), risk_head에 z_rna를 직결 concat하는 경로만 떼어내 별도의 고전적
        # Cox 가산항(rna_linear, zero-init)으로 바꾼다. rna_gate_only(z_rna를 아예 안 씀)와는
        # 달리 z_rna의 marginal 기여를 구조적으로 분리해서 보존한다는 점이 다르다.
        self.rna_combine_mode = rna_combine_mode
        if rna_combine_mode == "cox_add":
            self.rna_linear = nn.Linear(rna_dim, 1, bias=False)
            nn.init.zeros_(self.rna_linear.weight)  # 초기엔 z_rna 가산항 없는 것과 동일
        self.use_margin = use_margin
        self.use_age_sex = use_age_sex
        self.use_staging = use_staging
        self.use_mutation = use_mutation
        # 2026-08-09: scripts/diagnose_pma_component_reliance.py의 zero-ablation 진단(4개 성분
        # 다 지워봐도 손해가 없었음)에 이어, 구조적으로 하나를 빼고 재학습하는 ablation용
        # (train.py --drop-component). 기본 None이면 기존과 완전히 동일(4개 다 사용).
        # top_frac(train.py --top-frac)도 같이 노출 — top-k 성분이 상위 10%라는 너무 작은
        # 부분집합이라 노이즈에 민감했을 수 있다는 가설(top-k를 지우면 internal이 올랐던 것과
        # 같은 방향)을 검증한다. top_frac을 키우면(예: 0.25) 같은 "attention 상위" 컨셉은
        # 유지하되 표본을 넓혀 노이즈를 줄일 수 있는지 본다.
        self.use_tile_risk_head = use_tile_risk_head
        self.attn_pool = MultiComponentPooling(cfg.embed_dim, exclude=drop_component, top_frac=top_frac,
                                                use_tile_risk_head=use_tile_risk_head)
        # 2026-09-05: Nystrom(oversmoothing 무죄로 확인됨, scripts/diagnose_nystrom_oversmoothing*.py)도
        # ABMIL(entropy~0.999로 붕괴, weight_decay 무관하게 gradient가 아예 안 닿는 dead module로
        # 확인됨, scripts/diagnose_abmil_attn_training.py) 둘 다 N->1 풀링을 제대로 못 한다는
        # 진단에 이어, ABMIL/Nystrom을 완전히 우회하는 대안 — 학습 전혀 없는 unsupervised 군집화
        # (data/fit_clusters_uni2native.py가 raw feature 공간에서 미리 계산해 둔 K=10 중심,
        # scripts/extract_cluster_exemplars.py로 사람이 눈으로 "종양/기질/..." 해석 확인됨)로
        # 패치 수만 개를 K개의 "그 슬라이드에 실제로 존재하는 조직 유형" 대표값으로 미리
        # 요약한다. 이 K개를 기존 4-component(mean/std/attn/top) 자리에 그대로 꽂아 RNA
        # co-attention(component_coattn, 이미 gradient가 살아있는 걸로 확인된 모듈)에 넘긴다 —
        # "어떤 조직 유형이 이 환자의 RNA subtype에 중요한가"를 co-attention이 직접 고르게 하는
        # 것으로, ABMIL이 실패한 "N개 중 중요한 것 찾기"를 gradient 경로가 훨씬 짧은 co-attention
        # 쪽으로 옮긴다.
        self.cluster_pool = cluster_pool
        # 2026-09-05(2차): cluster_pool 단독(external C 0.535->0.608, M7 대비 손해가 거의 사라짐)
        # 결과를 본 뒤, "원본 PMA(Nystrom+ABMIL+co-attn)에서 ABMIL만 cluster_pool로 갈아끼우면
        # 어떤지"도 확인 — Nystrom(패치 간 self-attention, oversmoothing 무죄로 이미 확인됨)은
        # 그대로 살려서 self.vit로 문맥화(ctx_tokens)까지 한 뒤, 그 ctx_tokens를 raw feature
        # 기반 군집 배정(assign은 원래 있는 raw-feature 공간에서 결정, ctx_tokens는 64차원이라
        # cluster_centroids와 직접 비교 불가)에 따라 평균 낸다 — "Nystrom이 문맥을 섞어준 뒤의
        # 표현"을 군집별로 요약하는 조합.
        self.cluster_pool_after_vit = cluster_pool_after_vit
        # 2026-09-05(3차): None(기본)이면 기존 hard argmin 그대로. 양수면 fuzzy soft assignment
        # 온도(작을수록 hard에 가까움, 클수록 균등에 가까움) — raw feature 공간에서 k-means
        # inertia 실측(TCGA-only, K=11 재적합 기준 patch당 평균 제곱거리 ~190, RMS거리 ~13.8)
        # 대비 적당히 부드럽게 걸치도록 register 시점에 스케일 감 잡을 것.
        self.cluster_pool_temperature = cluster_pool_temperature
        if cluster_pool:
            path = cluster_centroids_path or f"data/cluster_centroids_{backbone}.pt"
            centroids = torch.load(path, weights_only=True)
            self.register_buffer("cluster_centroids", centroids.float())
        # 2026-09-05(4차): 비지도 k-means의 silhouette가 매우 낮았던 것(0.02~0.04, 종양/정상
        # 조직을 깨끗이 못 가름)을 보완 — PanNuke(핵 단위 라벨, 우리 코호트/라벨 완전 미참조,
        # scripts/train_hdp_pretrain_head.py)로 학습된 frozen TumorContentHead(models/
        # tumor_content_head.py)의 패치별 종양함량 점수(0~1)를 군집 가중치에 곱한다 — 완전히
        # 새로운 지도학습이 아니라 이미 있던, 우리 라벨과 무관한 외부 학습 자산을 재활용.
        self.tumor_content_head = None
        if tumor_content_head_path is not None:
            ckpt = torch.load(tumor_content_head_path, map_location="cpu", weights_only=False)
            head = TumorContentHead(in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"])
            head.load_state_dict(ckpt["state_dict"])
            for p in head.parameters():
                p.requires_grad = False
            head.eval()
            self.tumor_content_head = head
        # 2026-08-31: co-attention이 WSI가 성능에 안 먹히는 원인 셋(Nystrom self-attn/ABMIL/
        # co-attention) 중 하나인지 분리 검증하는 ablation용(train.py --no-coattn). False면
        # component_coattn 자체를 안 만들고, combine_with_clinical_rna()가 RNA-query 가중합 대신
        # 4개 관점의 단순 평균(z_wsi = patient_embed.mean(dim=0))을 쓴다 — "RNA가 4개 관점 중
        # 뭘 볼지 고르는 게" 도움이 되는지 vs 그냥 다 균등하게 보는 것과 차이가 없는지 검증.
        self.use_coattn = use_coattn
        if use_coattn:
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
            # 2026-08-20: RNA cox_add와 대칭 맞추려고 ClinicalEncoder(MLP)를 거치게 한 번 바꿨으나
            # ablation 결과(M7 기준) internal이 -0.025 하락 — clinical 신호가 원래 약해 MLP를
            # 추가하면 오히려 과적합만 늘어난다는 걸 확인, raw feature 직결 방식으로 원복(사용자
            # 결정). RNA는 신호가 강해 인코더가 도움이 되지만 clinical은 반대라는 뜻.
            self.register_buffer("age_mean", torch.tensor(age_mean, dtype=torch.float32))
            self.register_buffer("age_std", torch.tensor(age_std, dtype=torch.float32))
            if use_margin:
                m_mean, m_std = margin_stats
                self.register_buffer("margin_mean", torch.tensor(m_mean, dtype=torch.float32))
                self.register_buffer("margin_std", torch.tensor(m_std, dtype=torch.float32))
            if use_staging:
                if stage_stats is None:
                    raise ValueError("use_staging=True면 stage_stats가 필요합니다.")
                for field in STAGE_FIELDS:
                    mean, std = stage_stats[field]
                    short = _STAGE_BUFFER_NAMES[field]
                    self.register_buffer(f"{short}_mean", torch.tensor(mean, dtype=torch.float32))
                    self.register_buffer(f"{short}_std", torch.tensor(std, dtype=torch.float32))
            if use_mutation:
                # models/vit_m4.py::ViT_M4와 동일 관례(PDAC 4대 driver gene mutation status,
                # 2026-09-06 이식) — margin/staging과 같은 known-indicator 방식.
                if mutation_stats is None:
                    raise ValueError("use_mutation=True면 mutation_stats가 필요합니다.")
                for field in MUTATION_FIELDS:
                    mean, std = mutation_stats[field]
                    short = _MUTATION_BUFFER_NAMES[field]
                    self.register_buffer(f"{short}_mean", torch.tensor(mean, dtype=torch.float32))
                    self.register_buffer(f"{short}_std", torch.tensor(std, dtype=torch.float32))
            raw_dim = ((2 if use_age_sex else 0) + (2 if use_margin else 0)
                       + (2 * len(STAGE_FIELDS) if use_staging else 0)
                       + (2 * len(MUTATION_FIELDS) if use_mutation else 0))
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
        # 2026-08-14: use_tile_risk_head=True면 MultiComponentPooling이 반환하는 risk_stats(10개
        # 스칼라, 레퍼런스 MorphologyBurdenPooling과 동일 정의 — models/multi_component_pooling.py
        # 참조)도 risk_head 입력에 spatial_feat과 나란히 추가한다.
        risk_stats_dim = 10 if use_tile_risk_head else 0
        # risk_head 입력: [z_wsi(WSI_D)] (+ z_clinical(clinical_dim), use_clinical=True이고
        # combine_mode="concat"일 때만 — cox_add는 clinical을 risk_head에 넣지 않고 최종
        # 스칼라에 별도로 더한다) (+ z_rna(rna_dim), rna_gate_only=False일 때만)
        # (+ spatial_feat(spatial_feat_dim), 위 두 플래그 중 하나라도 켜졌을 때만)
        # (+ risk_stats(risk_stats_dim), use_tile_risk_head=True일 때만)
        # (rna_dim/clinical_dim이 둘 다 기본값(None)이고 use_clinical=True, combine_mode="concat",
        # rna_gate_only=False, spatial_feat_dim=0, risk_stats_dim=0이면 3*cfg.embed_dim과 동일
        # — 기존 동작 보존)
        risk_input_dim = (
            cfg.embed_dim
            + (clinical_dim if (self.use_clinical and combine_mode == "concat") else 0)
            + (0 if (rna_gate_only or rna_combine_mode == "cox_add") else rna_dim)
            + spatial_feat_dim
            + risk_stats_dim
        )
        # surv_n_classes>1: train.py --surv-loss nll_surv(PORPOISE 원조 discretized-hazard NLL,
        # utils/losses.py::nll_surv_loss, 2026-09-06 이식) 전용 — models/vit_porpoise.py::
        # ViT_PORPOISE와 동일 관례. 기본값 1이면 기존 Cox 레시피와 완전히 동일.
        self.risk_head = nn.Sequential(
            nn.LayerNorm(risk_input_dim),
            nn.Linear(risk_input_dim, surv_n_classes),
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
        tumor_type: torch.Tensor | None = None,
    ) -> dict:
        if self.cluster_pool:
            if features is None:
                raise ValueError("cluster_pool=True는 precomputed features 모드에서만 지원합니다.")
            raw = features.to(coords.device, non_blocking=True).float()  # (N, raw_dim) — 투영 이전
            centroids = self.cluster_centroids.to(raw.device)             # (K, raw_dim)
            dist = torch.cdist(raw, centroids)                            # (N, K) — raw 공간 기준(고정)
            # 2026-09-05(3차): hard argmin(경계에 걸친 패치를 1개 군집에 확정 배정 — 경계 근처
            # 정보 손실) 대신, cluster_pool_temperature가 주어지면 -distance/T의 softmax로
            # 모든 군집에 부드럽게 걸치는 가중치를 쓴다(fuzzy c-means류). weights를
            # one-hot(hard)이든 softmax(soft)든 동일한 가중평균 수식으로 통일 — 분기 로직 중복 제거.
            if self.cluster_pool_temperature is not None:
                weights = torch.softmax(-dist / self.cluster_pool_temperature, dim=1)  # (N, K)
            else:
                assign = dist.argmin(dim=1)
                weights = torch.zeros_like(dist).scatter_(1, assign.unsqueeze(1), 1.0)  # (N, K) one-hot
            if self.tumor_content_head is not None:
                # 2026-09-05(4차): 비지도 k-means(silhouette 0.02~0.04로 매우 낮음 — 종양/정상
                # 조직을 깨끗하게 못 가름)를 보완 — PanNuke(핵 단위 라벨, 우리 코호트/라벨 완전
                # 미참조)로 학습된 frozen TumorContentHead의 패치별 종양함량 점수(0~1)를 군집
                # 가중치에 곱해, 같은 군집 안에서도 종양함량이 높은 패치가 그 군집 대표값에
                # 더 많이 기여하게 한다(정상/기질 조직이 대표값을 희석하는 것을 줄임).
                with torch.no_grad():
                    tumor_score = self.tumor_content_head(raw)  # (N,) 0~1
                weights = weights * tumor_score.unsqueeze(1)
            wsum = weights.sum(dim=0)                                      # (K,) — 군집별 유효 가중치 총합
            empty = wsum < 1e-6                                            # 이 배치에 사실상 배정 안 된 군집
            if self.cluster_pool_after_vit:
                # Nystrom(self.vit)까지 살려서 문맥화한 뒤(ctx_tokens, embed_dim) 그 표현을
                # 군집별로 가중평균 — 가중치는 위에서 이미 raw feature 공간으로 정해뒀다
                # (ctx_tokens는 embed_dim=64라 raw 1536차원 centroids와 직접 비교 불가).
                patch_tokens = self.cnn.forward_pooled(raw)                # (N, D)
                ctx_tokens = patch_tokens if self.skip_patch_vit else self.vit(patch_tokens, coords)  # (N, D)
                components = (weights.T @ ctx_tokens) / wsum.clamp(min=1e-8).unsqueeze(1)  # (K, D)
                components[empty] = ctx_tokens.mean(dim=0)
            else:
                cluster_raw = (weights.T @ raw) / wsum.clamp(min=1e-8).unsqueeze(1)  # (K, raw_dim)
                cluster_raw[empty] = centroids[empty]
                components = self.cnn.forward_pooled(cluster_raw)              # (K, D) — 기존 4관점 자리
            return {"embed": components, "meanpool_embed": components.mean(dim=0), "patch_tokens": components}
        patch_tokens = self._patch_tokens(coords, patch_paths, features, transform, chunk_size, tile_cache)
        if self.use_coord_embed:
            coord_input = coords[torch.randperm(coords.shape[0], device=coords.device)] if self.coord_embed_shuffle else coords
            pos = self.coord_embed(coord_input)  # (N, D)
            if self.coord_embed_concat:
                patch_tokens = self.coord_fusion(torch.cat([patch_tokens, pos], dim=-1))
            elif hasattr(self, "coord_embed_scale"):
                patch_tokens = patch_tokens + self.coord_embed_scale * pos
            else:
                patch_tokens = patch_tokens + pos
        if self.use_wsi_extra_mlp:
            patch_tokens = self.wsi_extra_mlp(patch_tokens)
        ctx_tokens = patch_tokens if self.skip_patch_vit else self.vit(patch_tokens, coords, tumor_type=tumor_type)
        components, attn_weights, risk_stats = self.attn_pool(ctx_tokens)  # (4, D), (N,), (10,) 또는 None
        # meanpool_embed: --rna-aux-weight(models/rna_predictor.py) 보조과제 입력 전용.
        # patch_tokens: scripts/train_spatial_residual.py(공간정보 잔차 branch) 전용 — Nystrom
        # *이전* raw CNN 출력을 그대로 노출해, 이미 근사/혼합된 ctx_tokens가 아니라 패치별
        # 독립적인 표현 위에서 kNN 그래프를 새로 구성할 수 있게 한다.
        out = {
            "embed": components, "attn_weights": attn_weights,
            "meanpool_embed": ctx_tokens.mean(dim=0), "patch_tokens": patch_tokens,
        }
        if risk_stats is not None:
            out["risk_stats"] = risk_stats
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
        risk_stats: torch.Tensor | None = None,  # (10,) — 환자 단위 평균, self.use_tile_risk_head=True일 때만
        mutation_ord: dict[str, torch.Tensor] | None = None,  # combine_mode="cox_add"에서만 의미 있음(아래서 안 씀)
    ) -> torch.Tensor:
        # mutation_ord는 여기서 안 쓰인다 — combine_mode="cox_add"(mutation 지원 조건, __init__
        # 검증 참조)에서는 clinical_kwargs 자체가 아래 concat 분기에서 쓰이지 않고, mutation은
        # train.py::_patient_risk가 별도로 self._clinical_embed(..., mutation_ord=...)를 호출해
        # 처리한다(margin_ord/stage_ord와 달리 raw feature Cox 가산항 경로). train.py가
        # use_mutation=True인 모델엔 항상 이 kwarg를 넘기므로 시그니처에서 받아만 준다.
        clinical_kwargs = {}
        if stage_ord is not None:
            clinical_kwargs["stage_ord"] = {k: v.unsqueeze(0) for k, v in stage_ord.items()}
        if margin_ord is not None:
            clinical_kwargs["margin_ord"] = margin_ord.unsqueeze(0)
        if self.use_coattn:
            z_wsi, _ = self.component_coattn(patient_embed, z_rna)  # (D,) — RNA가 4개 관점 중 골라 가중합
        else:
            z_wsi = patient_embed.mean(dim=0)  # (D,) — co-attention 없이 4개 관점 단순 평균
        parts = [z_wsi]
        if self.use_clinical and self.combine_mode == "concat":
            # combine_mode="cox_add"면 clinical은 여기서 임베딩/concat되지 않고, 호출부(train.py)가
            # risk_head(이 fused 임베딩) 계산 뒤 self.clinical_linear(raw)를 별도로 더한다.
            z_clinical = self.clinical_encoder(
                age_years.unsqueeze(0), sex_idx.unsqueeze(0), **clinical_kwargs
            ).squeeze(0)  # (D,)
            parts.append(z_clinical)
        if not self.rna_gate_only and self.rna_combine_mode != "cox_add":
            # rna_gate_only=True거나 rna_combine_mode="cox_add"면 z_rna는 위 co-attention의
            # query로만 관여하고 risk_head에는 직결 concat하지 않는다. rna_gate_only는 z_rna의
            # marginal 기여 자체를 차단하는 것이고, cox_add는 그 기여를 risk_head 밖의 별도
            # 가산항(rna_linear)으로 옮기는 것이라 목적이 다르다(호출부 train.py가 더함).
            parts.append(z_rna)
        fused = torch.cat(parts, dim=-1)
        if spatial_feat is not None:
            fused = torch.cat([fused, spatial_feat], dim=-1)
        if risk_stats is not None:
            fused = torch.cat([fused, risk_stats], dim=-1)
        return fused

    def _clinical_embed(self, age_years: torch.Tensor, sex_idx: torch.Tensor,
                         margin_ord: torch.Tensor | None = None,
                         stage_ord: dict[str, torch.Tensor] | None = None,
                         mutation_ord: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """combine_mode="cox_add" 전용. 2026-08-20: ClinicalEncoder(MLP) 경유 버전을 ablation
        검증 후 raw feature 직결로 원복(이름은 train.py 호출부 호환을 위해 _clinical_embed 유지,
        실제로는 (1, raw_dim) raw z-score를 반환). stage_ord: self.use_staging=True일 때만 필요.
        mutation_ord: self.use_mutation=True일 때만 필요(2026-09-06, models/vit_m4.py::ViT_M4와
        동일 관례 이식)."""
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
        if self.use_mutation:
            for field in MUTATION_FIELDS:
                short = _MUTATION_BUFFER_NAMES[field]
                ordv = mutation_ord[field].float()
                known = (ordv >= 0).float()
                mean = getattr(self, f"{short}_mean")
                std = getattr(self, f"{short}_std")
                z = torch.where(ordv >= 0, (ordv - mean) / std, torch.zeros_like(ordv))
                feats += [z, known]
        return torch.stack(feats, dim=-1).unsqueeze(0)  # (1, raw_dim)
