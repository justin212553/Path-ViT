"""
TCGA-PAAD / CPTAC-PDA WSI 생존(OS) 예측 학습 스크립트
태스크: 환자(case) 단위 OS(overall survival) risk score 회귀 — Cox Proportional Hazards
배치:   환자 1명이 보유한 모든 슬라이드 임베딩을 평균 풀링해 risk score 1개 산출.
        Cox loss는 위험집합(risk set) 비교를 위해 여러 환자를 한 minibatch(cox_batch_size)로
        묶어야 하므로, 그 minibatch가 찰 때마다 backward + optimizer.step()을 수행한다.
손실:   Cox partial negative log-likelihood (utils/losses.py::cox_ph_loss)
데이터: WSISurvivalDataset (data/dataset.py, --dataset {tcga,cptac,both})

검증:   case 단위 6:2:2 stratified split(train/val/test) — (dataset, OS_event) 조합별로
        seed 고정 셔플 후 배정한다(data/dataset.py::_stratified_case_split). val은 매 epoch
        모델 선택(best checkpoint)에, test는 학습이 끝난 뒤 그 best checkpoint로 딱 한 번만
        평가하는 held-out 성능 확인용이다(internal test). --dataset both를 쓰면 TCGA+CPTAC
        전체를 하나의 풀로 합쳐 이 방식으로 나눈다(코호트 비율도 stratify에 포함되므로 유지됨).

        --external 플래그를 주면, 학습에 전혀 쓰이지 않은 반대 코호트 전체(tcga↔cptac 자동
        선택)를 best checkpoint로 딱 한 번 평가하는 external test도 internal test와 함께
        수행한다(기본은 미사용). internal test는 같은
        코호트 내부의 held-out case라 배치 효과(기관/스캐너 차이)가 없는 반면, external
        test는 아예 다른 기관 코호트라 실제 일반화 성능(cross-dataset)을 더 엄격하게
        보여준다(check_domain_shift.py 참조).
지표:   c-index, hazard ratio(HR, 95% CI), log-rank p-value, time-dependent AUC(12/24/36개월)
        (utils/metrics.py::compute_survival_metrics, compute_time_dependent_auc).
        HR/log-rank p는 risk score 중앙값으로 저위험/고위험군을 나눠 계산한다.
"""
import argparse
import copy
import math
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.optim.swa_utils
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import Config
from data.dataset import (
    WSISurvivalDataset, CLINICAL_PATHS, pdac_subtype_gene_ids, literature_guided_gene_ids,
    resolve_tcga_only_rna_genes, pathway_category_gene_ids, literature_guided_gene_ids_intersection,
    pdac_consistency_gene_ids,
)
from data.patch_utils import (
    FEATURES_AUG_FILENAME, PATCH_TRANSFORM, PATCH_TRANSFORM_AUGMENTED,
    PATCH_TRANSFORM_AUGMENTED_CACHED, PATCH_TRANSFORM_AUGMENTED_CACHED_STRONGBLUR,
    PATCH_TRANSFORM_512, build_tile_cache,
)
from models import (
    ViT_M1, ViT_M1_AvgPool, ViT_M1_Pool, ViT_M2_Pool, LateFusionViT, ViT_M2, ViT_M4, ViT_M4_AvgPool, ViT_M4A, ViT_MCAT, ViT_PORPOISE, ViT_M4B,
    ViT_PM4, ViT_PMA, ViT_M4A_FF, ViT_M2_FF, ViT_PMA_FF, ClinicalOnly, RNAOnly, RNAOnlyExtend,
)
from models.rna_predictor import RNAPredictionHead
from models.stage_predictor import StagePredictionHead
from models.clinical_encoder import (
    age_stats_from_csv, STAGE_FIELDS, stage_stats_from_df, margin_stats_from_df,
    MUTATION_FIELDS, mutation_stats_from_df,
)
from data.fit_clusters import CENTROIDS_DIR
from utils import load_env, send_slack
from utils.losses import cox_ph_loss, nll_surv_loss, hazard_to_risk, fit_survival_bins, digitize_survival_time
from utils.metrics import compute_survival_metrics, compute_time_dependent_auc
from utils.sam import SAM


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_amp_ctx() -> torch.autocast:
    """A30 전용 bfloat16 autocast — bf16은 fp32와 지수 범위가 같아 loss scaling이 불필요하다."""
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _build_scheduler(optimizer, cfg):
    """Linear warmup → cosine decay (epoch 단위)."""
    total  = cfg.train.epochs
    warmup = cfg.train.warmup_epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _identity_collate(batch: list) -> list:
    """batch_size=1 전제 — DataLoader가 환자 1명의 슬라이드 리스트를 그대로 통과시키도록 함."""
    return batch[0]


# 2026-08-15: --clinical-lr-mult/--rna-lr-mult(optimizer param_group lr 분리)와
# --auto-branch-balance(매 스텝 gradient norm 실시간 보정) 둘 다 같은 브랜치 정의를 공유한다.
_BRANCH_ATTRS = {
    "clinical": ("clinical_encoder", "clinical_linear"),
    "rna": ("rna_encoder", "rna_linear"),
    # 2026-09-03: --wsi-lr-mult용 — --sam-wsi-only가 이미 쓰던 WSI 브랜치 정의
    # (_WSI_BRANCH_ATTRS, 아래 --sam-wsi-only 분기)와 동일한 attribute 목록을 재사용.
    "wsi": ("cnn", "vit", "attn_pool", "multi_pool", "component_coattn", "dispersion_scale"),
}


def _attn_entropy(p: torch.Tensor) -> torch.Tensor:
    """p: (N,) 합=1인 attention 분포. 정규화 엔트로피(0~1, 1=완전균등) — 미분 가능(entropy
    정규화 loss, --entropy-reg-weight에서 씀). N<=1이면 정의상 0(엔트로피 없음)을 반환."""
    n = p.shape[0]
    if n <= 1:
        return p.new_zeros(())
    ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum()
    return ent / torch.log(torch.tensor(float(n), device=p.device))


def _branch_param_groups(model) -> dict[str, list]:
    """모델 파라미터를 clinical/rna/wsi/other 4개 그룹으로 나눈다(각 브랜치가 없는 모델이면 해당
    리스트는 빈 리스트)."""
    groups: dict[str, list] = {"clinical": [], "rna": [], "wsi": [], "other": []}
    attr_to_key = {attr: key for key, attrs in _BRANCH_ATTRS.items() for attr in attrs}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        top = name.split(".")[0]
        groups[attr_to_key.get(top, "other")].append(p)
    return groups


def _stage_ord_from_patient(patient_slides, device) -> dict[str, torch.Tensor] | None:
    """patient_slides[0]에 STAGE_FIELDS(with_staging=True로 로드된 경우만)가 있으면 device로 옮겨
    {field: () 스칼라 long} dict로 반환한다 - "미상"은 -1(data/dataset.py 규약). with_staging=False로
    로드된 데이터셋(--clinical-staging/--stage-aux-weight 둘 다 미사용)이면 None."""
    p = patient_slides[0]
    if STAGE_FIELDS[0] not in p:
        return None
    return {f: p[f].to(device, non_blocking=True) for f in STAGE_FIELDS}


def _margin_ord_from_patient(patient_slides, device) -> torch.Tensor | None:
    """_stage_ord_from_patient과 동일한 관례의 margin(residual_disease, --clinical-margin) 버전 —
    with_margin=True로 로드된 데이터셋이 아니면 None."""
    p = patient_slides[0]
    if "margin_ord" not in p:
        return None
    return p["margin_ord"].to(device, non_blocking=True)


def _mutation_ord_from_patient(patient_slides, device) -> dict[str, torch.Tensor] | None:
    """_stage_ord_from_patient과 동일한 관례의 mutation(--clinical-mutation) 버전 — with_mutation=True로
    로드된 데이터셋이 아니면 None."""
    p = patient_slides[0]
    if MUTATION_FIELDS[0] not in p:
        return None
    return {f: p[f].to(device, non_blocking=True) for f in MUTATION_FIELDS}


def _patient_risk(
    model, patient_slides, device, amp_ctx, transform, chunk_size, patch_keep_frac: float = 1.0,
    shuffle_patches: bool = False, tile_cache: dict | None = None,
    patch_subsample_generator: torch.Generator | None = None,
    modality_dropout_p: float = 0.0,
    branch_risk_out: dict | None = None,
):
    """환자 1명이 보유한 슬라이드 전부를 forward해 임베딩을 평균 풀링한 뒤 risk score(scalar)를 계산한다.
    Returns: (risk, aux_loss, stage_aux_loss) — aux_loss는 model.rna_aux_head가 있을 때만 텐서
    (--rna-aux-weight, models/rna_predictor.py 참조), stage_aux_loss는 model.stage_aux_head가
    있을 때만 텐서(--stage-aux-weight, models/stage_predictor.py 참조), 둘 다 없으면 None.

    branch_risk_out: --ogm-ge 전용(2026-08-15)이었으나 2026-08-31 --entropy-reg-weight도
    같은 side-channel을 공유하도록 확장. dict를 넘기면 combine_mode="cox_add"일 때 WSI 단독
    항("wsi", clinical/rna cox_add 가산 *전* risk_head 출력)과 clinical 단독 항("clinical",
    model.clinical_linear 출력), 그리고 attn_pool의 슬라이드 평균 patch attention entropy
    ("attn_entropy", 0~1 정규화, model이 attn_weights를 반환하는 WSI 모델일 때만)를 이 dict에
    곁다리로 채워넣는다(반환값 자체는 안 바뀜 — 기존 호출부 전부와 호환). None이면 아무 동작
    안 함(기본값, 부작용 없음).

    [patch_keep_frac, --patch-keep-frac(PatchDropout)] model.training(=True, train_one_epoch에서
    호출될 때만)일 때만 슬라이드 패치를 이 비율만큼 랜덤 서브샘플한다 — val/test/external
    평가(evaluate(), model.eval())에서는 항상 전체 패치를 그대로 쓴다(평가 지표 안정성 유지).
    mean/std/attn-weighted/top-k pooling은 전부 N에 대해 이미 정규화돼 있어 별도 스케일
    보정 없이 인덱스만 서브셋으로 잘라도 된다(findings_backlog.md 7번 항목).

    [--M2/--M4/--M4A/--M4B] model이 clinical_encoder(및 rna_encoder)를 보유하면, age/sex(/rna)는
    슬라이드가 아니라 환자 단위 메타데이터이므로 슬라이드 평균 풀링 이후
    combine_with_clinical()(--M2) 또는 combine_with_clinical_rna()(--M4/--M4A/--M4B)로 결합한다.

    [--M4, RNA-guided attention pooling] rna_encoder가 있으면 z_rna는 슬라이드 루프
    *이전에* 먼저 encode_rna()로 계산해, 각 슬라이드 forward(rna_context=z_rna)에 넘긴다 —
    ABMIL의 patch attention score 자체가 z_rna로 조건화되므로(vit_m1.py::AttentionPooling),
    풀링이 끝난 뒤에야 RNA를 아는 --M2 방식의 clinical 결합과 다르다. --M4B는 z_rna를
    attn_pool이 아니라 ViT 입력 토큰에 FiLM으로 적용하지만(vit_m4b.py), rna_context를
    forward에 넘기는 배선 자체는 --M4와 동일하다.

    [--M5/--M6] WSI가 전혀 없는 모델(model에 .cnn이 없음) — 슬라이드 순회 자체가
    불필요하다. Clinical 또는 RNA 중 하나만 보고 바로 risk score를 계산한다.

    [modality_dropout_p, --modality-dropout-p] 2026-08-11 — RNA가 있는 모델(hasattr(model,
    "rna_encoder"))에서만, model.training일 때 이 확률로 z_rna를 통째로 0벡터로 지운다.
    diagnose_wsi_gradients.py 진단(findings_backlog.md)에서 RNA 인코더 gradient norm이 학습
    내내 WSI의 ~4배였던 것 — RNA가 강한 신호라 risk_head가 RNA에 안주하고 WSI/clinical
    브랜치가 상대적으로 undertrained되는 "modality imbalance" 문제에 대한 대응이다(Peng et al.
    CVPR 2022 OGM-GE 등이 다루는 문제와 동일 계열, 여기서는 그중 가장 단순한 modality dropout
    방식을 쓴다). z_rna를 rna_true(--rna-aux-weight 보조 loss 타깃) 계산 *이후*에 지우므로
    보조 loss는 항상 진짜 RNA 값을 본다 — 지워지는 건 메인 risk 경로(co-attention query로도,
    concat/cox_add 항으로도 쓰이는 z_rna)뿐이다. 이 한 번의 재할당이 PMA(component_coattn
    query)와 M4/M4A(rna_context, combine_with_clinical_rna)의 z_rna 사용처 전부에 자연스럽게
    전파되므로, 모델 클래스 쪽 코드는 전혀 안 건드려도 된다.
    """
    if not hasattr(model, "cnn"):
        with amp_ctx:
            p = patient_slides[0]
            if hasattr(model, "rna_encoder"):
                rna = p["rna"].to(device, non_blocking=True)
                return model(rna), None, None
            age_years = p["age_years"].to(device, non_blocking=True)
            sex_idx   = p["sex_idx"].to(device, non_blocking=True)
            stage_kwargs = {}
            _clinical_enc_m5 = getattr(model, "clinical_encoder", None)
            if _clinical_enc_m5 is not None and _clinical_enc_m5.use_staging:
                stage_kwargs["stage_ord"] = _stage_ord_from_patient(patient_slides, device)
            if _clinical_enc_m5 is not None and getattr(_clinical_enc_m5, "use_margin", False):
                stage_kwargs["margin_ord"] = _margin_ord_from_patient(patient_slides, device)
            return model(age_years, sex_idx, **stage_kwargs), None, None

    with amp_ctx:
        z_rna = None
        rna_true = None
        if hasattr(model, "rna_encoder"):
            rna = patient_slides[0]["rna"].to(device, non_blocking=True)
            z_rna = model.encode_rna(rna)  # (D,)
            rna_true = rna
            if model.training and modality_dropout_p > 0 and torch.rand(()).item() < modality_dropout_p:
                z_rna = torch.zeros_like(z_rna)

        slide_embeds = []
        slide_meanpool_embeds = []
        slide_spatial_feats = []
        slide_risk_stats = []
        slide_attn_entropies = []
        for slide in patient_slides:
            coords = slide["coords"]
            features = slide.get("features")
            patch_paths = slide.get("patch_paths")

            # [--shuffle-patches] list_patch_paths()가 항상 좌표순 정렬된 고정 순서를 반환하므로,
            # NystromAttention의 landmark(순서대로 연속 그룹을 평균낸 근사, nystrom_attention
            # 패키지 참조)가 매 epoch 똑같은 그룹핑을 반복해왔다 — patch_keep_frac<1.0(랜덤
            # 서브샘플)일 땐 torch.randperm이 부수효과로 이미 순서도 섞어왔지만, frac=1.0이면
            # 이 효과가 전혀 없었다. shuffle_patches=True면 frac=1.0이어도(k=n) 순서만 순수하게
            # 매 epoch 다시 섞는다(findings_backlog.md 참조 — 정규화 효과 vs 나이스트롬 근사 품질
            # 저하 트레이드오프를 직접 검증하는 실험).
            if model.training and (patch_keep_frac < 1.0 or shuffle_patches):
                n = coords.shape[0]
                k = max(1, round(n * patch_keep_frac))
                idx = (
                    torch.randperm(n, generator=patch_subsample_generator)[:k]
                    if patch_subsample_generator is not None
                    else torch.randperm(n)[:k]
                )
                coords = coords[idx]
                if features is not None:
                    features = features[idx]
                if patch_paths is not None:
                    patch_paths = [patch_paths[i] for i in idx.tolist()]

            coords = coords.to(device, non_blocking=True)
            forward_kwargs = {"rna_context": z_rna} if z_rna is not None else {}
            if "tumor_status" in slide:
                forward_kwargs["tumor_type"] = slide["tumor_status"].to(device, non_blocking=True)
            if features is not None:
                out = model(coords, features=features, **forward_kwargs)
            else:
                out = model(coords, patch_paths=patch_paths, transform=transform,
                             chunk_size=chunk_size, tile_cache=tile_cache, **forward_kwargs)
            slide_embeds.append(out["embed"])
            if "meanpool_embed" in out:
                slide_meanpool_embeds.append(out["meanpool_embed"])
            if "spatial_feat" in out:
                slide_spatial_feats.append(out["spatial_feat"])
            if "risk_stats" in out:
                slide_risk_stats.append(out["risk_stats"])
            if branch_risk_out is not None and "attn_weights" in out:
                # [--entropy-reg-weight] attn_pool의 patch attention이 균등분포로 붕괴하는
                # 문제(findings_backlog.md — entropy 0.999+, 코호트 크기·나이스트롬 유무와
                # 무관하게 재현)를 학습 중에 직접 벌점으로 억제해보는 ablation. branch_risk_out
                # (--ogm-ge와 동일한 기존 side-channel, 새 반환값 안 늘림)에 슬라이드 평균
                # entropy를 얹어 train_one_epoch이 꺼내 쓰게 한다.
                slide_attn_entropies.append(_attn_entropy(out["attn_weights"]))

        if branch_risk_out is not None and slide_attn_entropies:
            branch_risk_out["attn_entropy"] = torch.stack(slide_attn_entropies).mean()

        patient_embed = torch.stack(slide_embeds).mean(dim=0)      # (D,) — 슬라이드 평균 풀링
        patient_spatial_feat = torch.stack(slide_spatial_feats).mean(dim=0) if slide_spatial_feats else None
        patient_risk_stats = torch.stack(slide_risk_stats).mean(dim=0) if slide_risk_stats else None

        patient_meanpool = None
        if slide_meanpool_embeds and (hasattr(model, "rna_aux_head") or hasattr(model, "stage_aux_head")
                                       or hasattr(model, "clinical_aux_head")):
            patient_meanpool = torch.stack(slide_meanpool_embeds).mean(dim=0)  # (D,) — RNA/clinical-free

        aux_loss = None
        if hasattr(model, "rna_aux_head") and patient_meanpool is not None:
            rna_pred = model.rna_aux_head(patient_meanpool)
            aux_loss = F.mse_loss(rna_pred, rna_true)

        stage_aux_loss = None
        if hasattr(model, "stage_aux_head") and patient_meanpool is not None:
            stage_ord = _stage_ord_from_patient(patient_slides, device)
            stage_aux_loss = model.stage_aux_head.loss(
                patient_meanpool, stage_ord["ajcc_t"], stage_ord["tumor_grade"]
            )

        # 2026-09-04: models/clinical_aux_classifier.py::ClinicalAuxClassifier(scripts/experiment_
        # cptac_clinical_aux.py 전용) — stage_aux_head와 같은 원리지만 CPTAC 전용 라벨(PNI/면역
        # 침윤)까지 필요해 이 함수가 모르는 값이다. branch_risk_out을 입력 side-channel로도 써서
        # (기존엔 출력 전용) 호출부가 미리 채워둔 pni_ord/immune_ord를 여기서 읽는다 — 둘 다 없으면
        # (branch_risk_out 자체가 없거나, 이 모델이 clinical_aux_head가 없으면) 완전히 비활성.
        if hasattr(model, "clinical_aux_head") and patient_meanpool is not None and branch_risk_out is not None:
            pni_ord = branch_risk_out.get("pni_ord")
            immune_ord = branch_risk_out.get("immune_ord")
            if pni_ord is not None and immune_ord is not None:
                stage_ord = _stage_ord_from_patient(patient_slides, device)
                branch_risk_out["clinical_aux_loss"] = model.clinical_aux_head.loss(
                    patient_meanpool, stage_ord["ajcc_t"], pni_ord, immune_ord
                )

        if hasattr(model, "combine_with_clinical_rna"):
            age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
            sex_idx   = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
            # --no-clinical(ViT_PMA use_clinical=False)이면 clinical_encoder 자체가 없다 —
            # getattr로 존재 여부부터 확인해야 한다(2026-07-29, --no-clinical 첫 실사용 중 발견).
            _clinical_enc = getattr(model, "clinical_encoder", None)
            # 2026-08-06: stage_ord도 margin_ord와 같은 버그(cox_add엔 clinical_encoder가 없어
            # _clinical_enc.use_staging만 보면 항상 False) — model 최상위 use_staging을 우선 본다.
            stage_ord = (
                _stage_ord_from_patient(patient_slides, device)
                if getattr(model, "use_staging", False) or
                   (_clinical_enc is not None and getattr(_clinical_enc, "use_staging", False))
                else None
            )
            # 2026-08-05: model.combine_mode="cox_add"(ViT_PMA, train.py --combine-mode)면
            # clinical_encoder 자체가 없다 — models/clinical_rna_only.py에서 겪은 것과 같은 버그
            # (_clinical_enc.use_margin으로만 판단하면 cox_add에서 margin_ord가 항상 None이 됨)를
            # 피하려 model 최상위 use_margin(combine_mode 무관하게 항상 있음)을 우선 본다.
            margin_ord = (
                _margin_ord_from_patient(patient_slides, device)
                if getattr(model, "use_margin", False) or
                   (_clinical_enc is not None and getattr(_clinical_enc, "use_margin", False))
                else None
            )
            # risk_stats는 models/vit_pma.py::ViT_PMA(use_tile_risk_head=True)만 지원한다 —
            # M4/M4A/M4B의 combine_with_clinical_rna는 이 kwarg 자체가 없으므로, patient_risk_stats가
            # None일 때(=PMA가 아니거나 use_tile_risk_head=False)는 아예 안 넘겨 TypeError를 피한다.
            extra_kwargs = {"risk_stats": patient_risk_stats} if patient_risk_stats is not None else {}
            # 2026-09-03: mutation_ord도 risk_stats와 같은 이유(models/vit_m4.py::ViT_M4만
            # combine_with_clinical_rna가 mutation_ord kwarg를 받는다 — M4A/M4B/PM4/PMA는 아직
            # 없음)로 use_mutation=True인 모델(--M4 --clinical-mutation)에서만 넣는다.
            if getattr(model, "use_mutation", False):
                extra_kwargs["mutation_ord"] = _mutation_ord_from_patient(patient_slides, device)
            patient_embed = model.combine_with_clinical_rna(
                patient_embed, age_years, sex_idx, z_rna, stage_ord=stage_ord, margin_ord=margin_ord,
                spatial_feat=patient_spatial_feat, **extra_kwargs,
            )  # (3D,) (+ spatial_feat_dim, models/spatial_features.py 켜졌을 때만)
        elif hasattr(model, "combine_with_clinical_pool"):
            # models/vit_m2_pool.py::ViT_M2_Pool(--M2_POOL) — z_clinical을 4개 pooling 관점의
            # co-attention query로 쓴다(PMA가 z_rna를 쓰는 것과 대칭). combine_with_clinical_rna와
            # 달리 patient_embed가 (4,D) 성분 그대로 넘어온다(train.py가 슬라이드별 out["embed"]를
            # 평균 풀링한 것 — ViT_M2_Pool.forward가 PMA와 동일하게 4개 성분을 반환하기 때문).
            age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
            sex_idx   = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
            margin_ord = (
                _margin_ord_from_patient(patient_slides, device)
                if getattr(model, "use_margin", False) else None
            )
            # 2026-08-17: M2_POOL도 staging 지원 추가(models/vit_m2_pool.py) — combine_with_clinical
            # 분기(아래)와 동일 관례로 clinical_encoder.use_staging까지 함께 확인한다(margin_ord와
            # 같은 이유, clinical_encoder 자체가 없는 cox_add+selfattn 조합 대비).
            _clinical_enc = getattr(model, "clinical_encoder", None)
            stage_ord = (
                _stage_ord_from_patient(patient_slides, device)
                if getattr(model, "use_staging", False) or
                   (_clinical_enc is not None and getattr(_clinical_enc, "use_staging", False))
                else None
            )
            patient_embed = model.combine_with_clinical_pool(
                patient_embed, age_years, sex_idx, stage_ord=stage_ord, margin_ord=margin_ord,
                spatial_feat=patient_spatial_feat,
            )  # (2D,) (+ spatial_feat_dim). combine_mode="cox_add"면 (D,)/(D+1,) — clinical은
            # 여기서 안 섞이고 아래 공용 블록에서 더해짐(models/vit_m2_pool.py 2026-08-07).
        elif hasattr(model, "combine_with_clinical"):
            # --M2_FF: rna_encoder는 있지만(FFN 직전 FiLM용) 최종 결합엔 RNA를 직접 노출하지 않는
            # 모델이라, encoder 존재 여부가 아니라 결합 메서드 존재 여부로 분기해야 한다.
            age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
            sex_idx   = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
            # 2026-08-07: combine_mode="cox_add"(ViT_M2)면 clinical_encoder 자체가 없다 — ViT_PMA에서
            # 겪은 것과 같은 버그를 피하려 model 최상위 use_staging을 우선 본다.
            _clinical_enc = getattr(model, "clinical_encoder", None)
            stage_ord = (
                _stage_ord_from_patient(patient_slides, device)
                if getattr(model, "use_staging", False) or
                   (_clinical_enc is not None and getattr(_clinical_enc, "use_staging", False))
                else None
            )
            # 2026-08-14: M2에 margin(--clinical-margin) 지원 추가 — 그 전엔 M2엔 margin 입력이
            # 아예 없어 항상 None이었다(M2_FF 등 margin 미지원 모델은 getattr가 False로 안전
            # 폴백). model 최상위 use_margin을 우선 본다(cox_add면 clinical_encoder 자체가 없어
            # _clinical_enc.use_margin만 보면 항상 False가 되는 버그를 피하려는 것 — PMA/M4와 동일 관례).
            margin_ord = (
                _margin_ord_from_patient(patient_slides, device)
                if getattr(model, "use_margin", False) or
                   (_clinical_enc is not None and getattr(_clinical_enc, "use_margin", False))
                else None
            )
            patient_embed = model.combine_with_clinical(
                patient_embed, age_years, sex_idx, stage_ord=stage_ord, margin_ord=margin_ord,
                spatial_feat=patient_spatial_feat,
            )  # (2D,) (+ spatial_feat_dim, 2026-07-30 — M1/M2에도 dispersion 확장, train_multi.py)
            # (combine_mode="cox_add"면 (D,)/(D+1,) — clinical은 위에서 안 섞이고 아래 공용 블록에서 더해짐)
        elif hasattr(model, "pool_components"):
            # models/vit_m1_pool.py::ViT_M1_Pool(--M1_POOL) — 외부 모달리티 없이 학습된 고정
            # query로 4개 pooling 관점을 co-attention한다. patient_embed는 (4,D) 성분 그대로.
            patient_embed = model.pool_components(patient_embed)
            if patient_spatial_feat is not None:
                patient_embed = torch.cat([patient_embed, patient_spatial_feat], dim=-1)
        elif patient_spatial_feat is not None:
            # 2026-07-30: M1(ViT_M1)은 combine 메서드가 없어 patient_embed에 직접 이어붙인다 —
            # use_attn_dispersion=True인 M1을 이전엔 아무도 안 써서 드러나지 않았던 버그
            # (train_multi.py에서 M1/M2에도 dispersion을 확장하며 발견, RuntimeError:
            # LayerNorm 차원 불일치).
            patient_embed = torch.cat([patient_embed, patient_spatial_feat], dim=-1)

        # --surv-loss nll_surv(models/vit_porpoise.py::ViT_PORPOISE surv_n_classes>1)면 risk_head가
        # (n_bins,) raw hazard logit을 뱉는다 — 아래 cox_add 가산은 브로드캐스팅으로 각 구간에
        # 동일하게 적용되고(PH 모델의 공변량 가산과 같은 원리), 함수 끝에서 스칼라로 변환한다.
        risk = model.risk_head(patient_embed.unsqueeze(0)).view(-1)  # (1,) 또는 (n_bins,)
        if branch_risk_out is not None:
            branch_risk_out["wsi"] = risk
        if getattr(model, "combine_mode", "concat") == "cox_add":
            # models/vit_pma.py::ViT_PMA combine_mode="cox_add" — clinical은 위 patient_embed에
            # 안 섞여 있고(combine_with_clinical_rna가 concat 안 함), 여기서 고전적 Cox 가산항으로
            # 최종 risk 스칼라에 직접 더한다(models/clinical_rna_only.py::ClinicalRNAOnly와 동일 관례).
            # 2026-09-03: mutation_ord도 combine_with_clinical_rna의 extra_kwargs와 동일한 이유로
            # use_mutation=True인 모델(models/vit_m4.py::ViT_M4._clinical_embed만 이 kwarg를 받음)
            # 에서만 넣는다 — 다른 cox_add 모델(PMA 등)의 _clinical_embed는 mutation_ord 자체가 없다.
            clinical_embed_kwargs = (
                {"mutation_ord": _mutation_ord_from_patient(patient_slides, device)}
                if getattr(model, "use_mutation", False) else {}
            )
            clin_embed = model._clinical_embed(age_years, sex_idx, margin_ord, stage_ord=stage_ord,
                                                **clinical_embed_kwargs)
            clinical_term = model.clinical_linear(clin_embed).view(1)
            if branch_risk_out is not None:
                branch_risk_out["clinical"] = clinical_term
            risk = risk + clinical_term
        if getattr(model, "rna_combine_mode", "concat") == "cox_add":
            # 2026-08-09: models/vit_pma.py::ViT_PMA rna_combine_mode="cox_add" — z_rna는 여전히
            # component_coattn의 query로 WSI pooling을 guide했지만(combine_with_clinical_rna에서
            # 이미 반영됨), risk_head에는 직결 concat되지 않았으므로 여기서 별도 Cox 가산항으로
            # 더한다. clinical cox_add와 완전히 같은 관례 — z_rna는 위에서 이미 계산돼 있음.
            rna_term = model.rna_linear(z_rna).view(1)
            if branch_risk_out is not None:
                branch_risk_out["rna"] = rna_term
            risk = risk + rna_term
    if risk.numel() > 1:
        # nll_surv 모드 — 학습 loss에 쓸 raw hazard logit을 옆으로 빼두고(train_one_epoch만 읽음,
        # evaluate()는 아래 스칼라 변환 결과만 받으므로 C-index/checkpoint/external eval 등
        # 기존 스칼라 risk 파이프라인은 전혀 안 건드려도 된다), 반환값 자체는 항상 스칼라로
        # 맞춘다(utils/losses.py::hazard_to_risk).
        if branch_risk_out is not None:
            branch_risk_out["hazard_logits"] = risk
        risk = hazard_to_risk(risk).view(1)
    return risk, aux_loss, stage_aux_loss


def train_one_epoch(
    model, loader, optimizer, cfg, device, amp_ctx, transform,
    patch_keep_frac: float = 1.0, rna_aux_weight: float = 0.0, stage_aux_weight: float = 0.0,
    shuffle_patches: bool = False, tile_cache: dict | None = None,
    patch_subsample_generator: torch.Generator | None = None,
    modality_dropout_p: float = 0.0,
    branch_groups: dict[str, list] | None = None,
    auto_balance_enabled: bool = False,
    ogm_ge_alpha: float | None = None,
    ogm_ge_epoch_progress: float = 0.0,
    entropy_reg_weight: float = 0.0,
    surv_loss: str = "cox",
    nll_bin_edges: np.ndarray | None = None,
    nll_cox_weight: float = 1.0,
    desc: str = "train",
) -> float:
    model.train()
    if hasattr(model, "cnn") and model.cnn.backbone is not None:
        model.cnn.backbone.eval()  # frozen backbone의 BN을 population stats(eval)로 고정 — train/eval 분포 불일치 방지
    total_loss    = 0.0
    total_batches = 0
    chunk_size    = cfg.train.cnn_chunk_size
    batch_size    = cfg.train.cox_batch_size

    risks, times, events, aux_losses, stage_aux_losses, patient_batch = [], [], [], [], [], []
    ogm_risks_wsi, ogm_risks_clinical = [], []
    entropy_losses = []
    is_sam = isinstance(optimizer, SAM)

    def _compute_loss(risk_list, time_t, event_t, aux_list, stage_aux_list, entropy_list):
        if surv_loss in ("nll_surv", "both"):
            # risk_list의 각 원소는 _patient_risk가 branch_risk_out["hazard_logits"]로 빼둔
            # (n_bins,) raw hazard logit(스칼라 변환 전) — torch.cat이 아니라 torch.stack으로
            # (B, n_bins)를 만든다. y(시간-구간 label)는 이 fold의 train split에서 미리 fit한
            # nll_bin_edges로 그때그때 계산한다(--nll-n-bins, train.py 메인 흐름 참조).
            h = torch.stack(risk_list)
            y_np = digitize_survival_time(time_t.detach().cpu().numpy(), nll_bin_edges)
            y = torch.from_numpy(y_np).to(h.device)
            loss = nll_surv_loss(h, y, event_t)
            if surv_loss == "both":
                # --nll-cox-weight: 같은 hazard logit에서 유도한 스칼라 risk(utils/losses.py::
                # hazard_to_risk)에 cox_ph_loss를 추가로 적용해 더한다 — 두 loss가 서로 다른
                # 것(전체 우도 vs 순위/분리력)을 최적화한다는 관찰에서, 한쪽만 고르지 않고
                # 같이 최적화해보는 ablation.
                loss = loss + nll_cox_weight * cox_ph_loss(hazard_to_risk(h), time_t, event_t)
        else:
            loss = cox_ph_loss(torch.cat(risk_list), time_t, event_t)
        if rna_aux_weight > 0 and aux_list:
            # --rna-aux-weight(models/rna_predictor.py): WSI 표현이 RNA 발현도 예측하도록
            # 보조 loss를 더한다 — 생존 라벨(환자당 1개, censoring으로 더 약함)만으로
            # 62만 파라미터짜리 WSI 브랜치를 학습시키는 게 병목이라는 진단(model_zoo.md)에 대한
            # 대응. 결합 방식이 아니라 학습 신호 자체를 보강한다.
            loss = loss + rna_aux_weight * torch.stack(aux_list).mean()
        if stage_aux_weight > 0 and stage_aux_list:
            # --stage-aux-weight(models/stage_predictor.py): 위와 동일 원리, 타깃만 T-stage/grade.
            loss = loss + stage_aux_weight * torch.stack(stage_aux_list).mean()
        if entropy_reg_weight > 0 and entropy_list:
            # --entropy-reg-weight: attn_pool의 patch attention entropy(0~1, 1=완전균등)를
            # loss에 직접 벌점으로 더한다 — 낮출수록(더 뾰족해질수록) loss가 줄어드니, Cox loss와
            # 함께 최소화하는 과정에서 attention이 균등분포로 붕괴하지 않도록 명시적으로 유도한다.
            # 2026-08-31: 이미 학습된 체크포인트에 재학습 없이 temperature만 후처리로 낮추는
            # sharpening은 오히려 성능을 떨어뜨렸다(T=1로 학습된 raw score를 T<1로 재해석하면
            # 신호뿐 아니라 노이즈까지 같이 증폭됨) — 그래서 이번엔 처음부터 이 벌점을 알고
            # 학습하게 한다.
            loss = loss + entropy_reg_weight * torch.stack(entropy_list).mean()
        return loss

    def _refresh_batch(patients):
        """SAM 2nd pass용 — perturb된 가중치로 같은 배치(환자 리스트)를 다시 forward한다."""
        risks2, aux2, stage_aux2, entropy2 = [], [], [], []
        for ps in patients:
            branch_risk_out2 = {} if (ogm_ge_alpha is not None or entropy_reg_weight > 0
                                       or surv_loss in ("nll_surv", "both")) else None
            r2, a2_, s2_ = _patient_risk(
                model, ps, device, amp_ctx, transform, chunk_size, patch_keep_frac,
                shuffle_patches=shuffle_patches, tile_cache=tile_cache,
                patch_subsample_generator=patch_subsample_generator,
                modality_dropout_p=modality_dropout_p,
                branch_risk_out=branch_risk_out2,
            )
            risks2.append(branch_risk_out2["hazard_logits"] if surv_loss in ("nll_surv", "both") else r2)
            if a2_ is not None:
                aux2.append(a2_)
            if s2_ is not None:
                stage_aux2.append(s2_)
            if branch_risk_out2 and "attn_entropy" in branch_risk_out2:
                entropy2.append(branch_risk_out2["attn_entropy"])
        return risks2, aux2, stage_aux2, entropy2

    def _rescale_branch_grads():
        """--auto-branch-balance: clinical/rna 브랜치의 gradient norm을 나머지(WSI 등) 파라미터의
        gradient norm에 맞춰 매 스텝 실시간으로 재조정한다 — --clinical-lr-mult/--rna-lr-mult가
        고정 배율을 학습 시작 전에 손으로 정하는 것과 달리, 그 스텝의 실제 gradient 크기 차이에
        맞춰 배율이 매번 달라진다."""
        if not auto_balance_enabled or branch_groups is None:
            return

        def _grad_norm(params):
            sq = sum(p.grad.norm().item() ** 2 for p in params if p.grad is not None)
            return sq ** 0.5

        # 2026-09-03: --wsi-lr-mult 추가로 _branch_param_groups가 "wsi"를 "other"에서 분리해냈다
        # (전엔 WSI(cnn/vit/attn_pool 등)가 "other"에 섞여 있었음) — 이 함수의 원래 의도("clinical/
        # rna를 WSI+나머지 전체 기준으로 맞춘다")가 안 바뀌게 여기서 다시 합쳐서 기준으로 쓴다.
        ref_norm = _grad_norm(branch_groups["other"] + branch_groups["wsi"])
        if ref_norm <= 0:
            return
        for key in ("clinical", "rna"):
            params = branch_groups[key]
            if not params:
                continue
            branch_norm = _grad_norm(params)
            if branch_norm <= 1e-8:
                continue
            scale = min(max(ref_norm / branch_norm, 0.1), 100.0)
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(scale)

    def _ogm_ge_modulate():
        """--ogm-ge: Peng et al. CVPR 2022(OGM-GE, "Balanced Multimodal Learning via On-the-fly
        Gradient Modulation")를 우리 Cox 구조에 맞게 이식한 근사 버전 — 원논문은 task별로
        별도 loss가 있는 세팅을 전제하는데, 우리는 Cox loss 하나뿐이라 논문의 정확한 수식을
        그대로 쓸 수 없다. 대신 combine_mode="cox_add"에서 WSI 항(risk_head 출력, cox_add
        가산 *전*)과 clinical 항(clinical_linear 출력)이 깨끗하게 분리되는 지점(_patient_risk의
        branch_risk_out)을 이용해, 이번 배치에서 각 브랜치 단독으로 순위를 얼마나 잘
        설명하는지(-Cox loss, 클수록 잘 맞음)를 "판별력 점수"로 삼는다. 더 잘 맞히는(=이미
        앞서가는) 쪽의 gradient를 tanh 계수로 억제하고(GradNorm/PCGrad와 달리 별도 task loss나
        공유 파라미터 위 gradient 충돌 없이도 성립), 억제한 만큼 작은 gaussian 노이즈를 더해
        (GE, epoch가 진행될수록 anneal) 그 브랜치가 이 배치의 우연에 안주하지 않게 한다.
        RNA는 기본 레시피(FiLM guided attention)에서 WSI pooling에 얽혀 있어 이 방식으로
        분리되지 않으므로(--rna-combine-mode cox_add일 때만 분리됨) 이 파일럿에선 다루지 않는다.
        """
        if ogm_ge_alpha is None or branch_groups is None or not ogm_risks_wsi or not ogm_risks_clinical:
            return
        with torch.no_grad():
            risk_wsi_batch = torch.cat(ogm_risks_wsi)
            risk_clin_batch = torch.cat(ogm_risks_clinical)
            time_t = torch.cat(times).to(device)
            event_t = torch.cat(events).to(device)
            score_wsi = -cox_ph_loss(risk_wsi_batch, time_t, event_t).item()
            score_clin = -cox_ph_loss(risk_clin_batch, time_t, event_t).item()
            if not (math.isfinite(score_wsi) and math.isfinite(score_clin)):
                return

            diff = score_wsi - score_clin
            dominant_key = "other" if diff > 0 else "clinical"
            k = max(1.0 - math.tanh(ogm_ge_alpha * abs(diff)), 0.0)

            # 2026-09-03: ref_norm과 동일한 이유 — "other"가 더 이상 WSI를 포함하지 않으므로
            # dominant_key=="other"(WSI 쪽이 우세)일 때는 "wsi"까지 합쳐서 억제한다.
            dominant_params = (
                branch_groups["other"] + branch_groups["wsi"] if dominant_key == "other"
                else branch_groups[dominant_key]
            )
            for p in dominant_params:
                if p.grad is not None:
                    p.grad.mul_(k)

            noise_scale = (1.0 - k) * (1.0 - ogm_ge_epoch_progress) * 0.01
            if noise_scale > 0:
                for p in dominant_params:
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * noise_scale * p.grad.std().clamp(min=1e-8))

    def _flush():
        nonlocal risks, times, events, aux_losses, stage_aux_losses, patient_batch, total_loss, total_batches
        nonlocal ogm_risks_wsi, ogm_risks_clinical, entropy_losses
        if not risks:
            return
        time_t  = torch.cat(times).to(device)
        event_t = torch.cat(events).to(device)

        loss = _compute_loss(risks, time_t, event_t, aux_losses, stage_aux_losses, entropy_losses)
        optimizer.zero_grad()
        loss.backward()
        _rescale_branch_grads()
        _ogm_ge_modulate()

        # 2026-07-26: cptac 방향 학습(dispersion 병행) 중 특정 배치에서 loss가 NaN으로 발산해
        # 그 순간부터 가중치 전체가 영구히 오염되는 현상 발견(RNA/clinical 원본 데이터엔 NaN/Inf
        # 없음 확인됨 — bf16 AMP 하에서 특정 risk set 조합이 유발하는 forward-pass 수치 불안정으로
        # 추정). gradient clipping은 gradient 자체가 NaN이면 무력하므로, step 직전에 loss/grad_norm이
        # non-finite인 배치는 파라미터 업데이트를 건너뛴다 — 그 배치 하나만 버리고 학습은 계속된다.
        if is_sam:
            # [SAM] 1st pass gradient로 근방 최악점까지 이동한 뒤, 같은 배치를 그 지점에서
            # 재평가(2nd pass)한 gradient로 실제 업데이트한다(utils/sam.py).
            optimizer.first_step()
            optimizer.zero_grad()
            risks2, aux2, stage_aux2, entropy2 = _refresh_batch(patient_batch)
            loss2 = _compute_loss(risks2, time_t, event_t, aux2, stage_aux2, entropy2)
            loss2.backward()
            _rescale_branch_grads()
            _ogm_ge_modulate()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not (torch.isfinite(loss2) and torch.isfinite(grad_norm)):
                print(f"  [경고] non-finite loss2(={loss2.item()})/grad_norm(={grad_norm.item()}) 배치 스킵(SAM)")
                optimizer.second_step(skip_update=True)
            else:
                optimizer.second_step(skip_update=False)
                total_loss    += loss2.item()
                total_batches += 1
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not (torch.isfinite(loss) and torch.isfinite(grad_norm)):
                print(f"  [경고] non-finite loss(={loss.item()})/grad_norm(={grad_norm.item()}) 배치 스킵")
                optimizer.zero_grad()
            else:
                optimizer.step()
                total_loss    += loss.item()
                total_batches += 1
        risks, times, events, aux_losses, stage_aux_losses, patient_batch = [], [], [], [], [], []
        ogm_risks_wsi, ogm_risks_clinical = [], []
        entropy_losses = []

    # mininterval=30: data/patch_utils.py::build_tile_cache와 동일한 이유(비-TTY 로그 폭주 방지).
    for patient_slides in tqdm(loader, desc=desc, unit="patient", mininterval=30):  # 환자 1명 분량의 슬라이드 리스트
        if len(patient_slides) == 0:
            continue
        branch_risk_out = {} if (ogm_ge_alpha is not None or entropy_reg_weight > 0
                                  or surv_loss in ("nll_surv", "both")) else None
        risk, aux_loss, stage_aux_loss = _patient_risk(
            model, patient_slides, device, amp_ctx, transform, chunk_size, patch_keep_frac,
            shuffle_patches=shuffle_patches, tile_cache=tile_cache,
            patch_subsample_generator=patch_subsample_generator,
            modality_dropout_p=modality_dropout_p,
            branch_risk_out=branch_risk_out,
        )

        risks.append(branch_risk_out["hazard_logits"] if surv_loss in ("nll_surv", "both") else risk)
        times.append(patient_slides[0]["OS_time"])
        events.append(patient_slides[0]["OS_event"])
        if aux_loss is not None:
            aux_losses.append(aux_loss)
        if stage_aux_loss is not None:
            stage_aux_losses.append(stage_aux_loss)
        if branch_risk_out and "wsi" in branch_risk_out and "clinical" in branch_risk_out:
            ogm_risks_wsi.append(branch_risk_out["wsi"].detach())
            ogm_risks_clinical.append(branch_risk_out["clinical"].detach())
        if branch_risk_out and "attn_entropy" in branch_risk_out:
            entropy_losses.append(branch_risk_out["attn_entropy"])
        if is_sam:
            patient_batch.append(patient_slides)

        if len(risks) >= batch_size:
            _flush()

    _flush()  # 마지막 남은 partial batch

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, cfg, device, amp_ctx, transform, tile_cache: dict | None = None,
             desc: str = "eval") -> dict:
    """tile_cache: --tile-augment --image 모드에서, 이 loader가 순회하는 patient들의 타일이 이미
    RAM에 캐싱돼 있으면(train_eval은 train_ds와 동일 데이터, val은 별도로 프리로드해둔 캐시)
    넘긴다 — 없으면(기본) 기존처럼 매 호출마다 디스크에서 그때그때 디코딩한다(test/external처럼
    학습 중 반복 안 되는 1회성 평가는 이대로가 맞다). 2026-08-04: train_eval/val을 매 epoch마다
    디스크에서 새로 디코딩하고 있던 게 HPC(느린 파일시스템)에서 epoch 1도 못 넘기는 병목이었다
    — train_eval은 train_ds의 tile_cache를 그대로 재사용하고, val은 작은 별도 캐시를 만들어
    넘기면 이 병목이 사라진다.
    """
    model.eval()
    all_risks, all_times, all_events, all_case_ids = [], [], [], []
    chunk_size = cfg.train.cnn_chunk_size

    for patient_slides in tqdm(loader, desc=desc, unit="patient", mininterval=30):
        if len(patient_slides) == 0:
            continue
        risk, _, _ = _patient_risk(model, patient_slides, device, amp_ctx, transform, chunk_size,
                                    tile_cache=tile_cache)

        all_risks.append(risk.float().item())
        all_times.append(float(patient_slides[0]["OS_time"].item()))
        all_events.append(int(patient_slides[0]["OS_event"].item()))
        all_case_ids.append(patient_slides[0]["case_id"])

    risks  = np.array(all_risks)
    times  = np.array(all_times)
    events = np.array(all_events)
    return {
        **compute_survival_metrics(risks, times, events),
        "risks": risks, "times": times, "events": events, "case_ids": all_case_ids,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="cptac", choices=["tcga", "cptac", "both"],
        help="OS 예측에 사용할 데이터셋 (기본: cptac). both면 TCGA+CPTAC 전체를 하나의 "
             "풀로 합쳐 train/val/test를 나눈다.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="cfg.data.seed / cfg.train.seed를 함께 덮어쓴다 (기본: config.py 값 그대로). "
             "case split 재현성과 학습 seed를 동시에 바꿔 여러 seed로 반복 실행할 때 쓴다.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="cfg.train.lr(기본 1e-5) 덮어쓰기. train_light.py --lr과 동일 관례 — WSI 스택은 "
             "이 값을 왜 1e-5로 낮게 잡았는지 재검토된 적이 없어서(findings_backlog.md, Ray "
             "Tune 스윕 보류 항목) 하이퍼파라미터 스윕용으로 노출한다. 기본값과 다르면 "
             "model_prefix에 _LR{lr}이 자동으로 붙는다.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=None,
        help="cfg.train.weight_decay(기본 1e-1) 덮어쓰기. 기본값과 다르면 model_prefix에 "
             "_WD{wd}가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=None,
        help="cfg.train.warmup_epochs(기본 3) 덮어쓰기. 기본값과 다르면 model_prefix에 "
             "_WARMUP{n}이 자동으로 붙는다.",
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="주어지면(0-based) internal train/val/test를 단일 6:2:2 대신 K-fold(data/dataset.py::"
             "_kfold_case_split)로 배정한다 — 이 fold를 test로, 나머지를 다시 60:20으로 train/val "
             "배정. fold=0..n_folds-1을 전부 돌려 test 예측을 pool_kfold_preds.py로 이어붙이면 "
             "internal 표본이 코호트 전체 크기가 된다(train_light.py --fold와 동일한 관례). --fold를 "
             "주면 예측을 .logs/kfold_preds/에 CSV로 저장한다. None(기본)이면 기존 단일 split.",
    )
    parser.add_argument("--n-folds", type=int, default=5, help="--fold와 함께 쓰는 전체 fold 개수.")
    parser.add_argument(
        "--group-ts", type=str, default=None,
        help="wandb Group 이름(<모델종류>_<group-ts>)에 쓸 타임스탬프(MMDD::HHMM 형식). "
             "여러 시드/코호트를 스윕하는 래퍼 스크립트가 첫 실행 전에 한 번 계산해 모든 "
             "python train.py 호출에 동일한 값을 넘기면, 그 세션에서 나온 같은 모델 종류의 "
             "모든 run(internal+external 전부)이 wandb에서 하나의 Group으로 묶인다. "
             "생략하면 이 실행 자체의 시작 시각을 써서 이 run 하나만의 그룹이 된다.",
    )
    parser.add_argument(
        "--rna-genes", type=str, default="subtype",
        choices=[
            "subtype", "literature_1000", "literature_1500", "literature_2000", "pathway8",
            "literature_500_tcga_only", "literature_1000_tcga_only",
            "literature_1500_tcga_only", "literature_fdr0.1_tcga_only",
            "literature_fdr0.1_cptac_only", "literature_1500_intersection",
            "pdac_consistency_500", "pdac_consistency_1000", "pdac_consistency_1500",
            "pdac_consistency_2000",
        ],
        help="RNA 브랜치(--M4/--M4A/--M4B/--PM4/--PMA/--M6/--M6X) 입력 유전자셋 선택. "
             "subtype(기본): pdac_subtype_gene_ids(), Bailey/Moffitt subtype 분류용 ~340개. "
             "literature_{1000,1500,2000}: data/select_rnaseq_genes.py 산출물 — 문헌 큐레이션 "
             "PDAC 유전자를 train split 내부 Cox score test 순위로 우선 배치하고 나머지를 "
             "Cox 순위로 채운, 생존 예측에 직접 최적화된 유전자셋(레퍼런스 방법론 이식). "
             "**주의**: 이 4개(subtype 제외 literature_*)는 TCGA+CPTAC train split을 Stouffer로 "
             "결합해 뽑은 것이라, --dataset tcga --external처럼 반대 코호트를 external test로 "
             "쓰는 실행에는 leakage가 있다(그 test case 중 상당수의 생존 라벨이 이미 유전자 "
             "선정에 쓰였음, findings_backlog.md) — --dataset both 비교에만 쓸 것. "
             "literature_{500,1500}_tcga_only: data/select_rnaseq_genes.py --single-cohort tcga "
             "산출물 — TCGA train split만 사용(CPTAC 데이터 자체를 로드하지 않음), Stouffer 결합 "
             "없음. --dataset tcga --external(TCGA로 학습 -> CPTAC 전체 external test) 전용, "
             "leakage 없음. 500은 같은 TCGA-only 순위에서 상위 500개만 자른 더 좁은 버전(EXT_500) "
             "— 1500 대비 입력 차원을 줄여 91명 train 표본 대비 과적합을 낮추려는 시도. "
             "pathway8: 개별 유전자 대신 문헌 큐레이션 8개 생물학적 카테고리의 평균 z-score "
             "(카테고리당 1개, 총 8차원) - SurvPath의 pathway token 방식. 표본 대비 차원을 "
             "크게 줄인다. 미리 `python -m data.select_rnaseq_genes`로 뽑아둬야 한다.",
    )
    parser.add_argument(
        "--patch-keep-frac", type=float, default=1.0,
        help="PatchDropout(패치 단위 서브샘플링, findings_backlog.md 7번 항목). 1.0(기본)이면 "
             "비활성 - 매 학습 epoch마다 슬라이드 패치를 이 비율만큼 랜덤 서브셋만 사용한다 "
             "(val/test/external 평가는 항상 전체 패치 사용, 지표 안정성 유지). WSI 모델(--M1 "
             "등 WSI를 쓰는 모든 --M*)에 적용 가능. 1.0 미만이면 wandb/checkpoint에 _SS 접미사가 "
             "자동으로 붙는다.",
    )
    parser.add_argument(
        "--patch-subsample-seed", type=int, default=None,
        help="2026-07-27: --patch-keep-frac의 랜덤 서브샘플(torch.randperm)을 --seed(모델 초기화/"
             "DataLoader 순서 등 나머지 전부)와 분리된 별도 torch.Generator로 고정한다. --full-train "
             "5시드 반복에서 순수 학습 노이즈(std~0.02)가 여전히 크게 남는 원인이 패치 서브샘플링 "
             "패턴 자체일 수 있다는 가설 검증용 — 예: --seed 168(초기화는 최악 시드 그대로) + "
             "--patch-subsample-seed 42(서브샘플 패턴만 최고 시드로 고정)로 서브샘플링만 격리해서 "
             "확인한다. 지정 안 하면(기본 None) 기존과 동일하게 전역 RNG를 그대로 쓴다.",
    )
    parser.add_argument(
        "--init-seed", type=int, default=None,
        help="2026-07-27: 모델 가중치 초기화만 --seed와 분리된 값으로 고정한다. torch.manual_seed()를 "
             "모델 생성 직전에 이 값으로, 생성 직후 다시 --seed로 되돌리는 방식(nn.Module 기본 "
             "reset_parameters()는 generator 인자를 안 받아 patch-subsample-seed처럼 완전히 독립된 "
             "Generator로는 분리할 수 없음 — 대신 재시딩 경계로 분리). --full-train 반복에서 split/"
             "dropout/patch-subsample 패턴/환자 순서를 전부 소거한 뒤 마지막으로 남은 후보(초기화 "
             "자체)를 격리 검증하기 위한 용도. 주의: 이후 DataLoader 셔플/dropout 등은 '원래 --seed "
             "런의 연속된 스트림'이 아니라 '모델 생성 직후 --seed로 새로 시작한 스트림'이라, 완전히 "
             "동일한 재현은 아니고 독립적으로 잘 정의된 비교군이라는 정도로 해석해야 한다.",
    )
    parser.add_argument(
        "--auc-days", type=str, default="365,730,1095",
        help="2026-07-28: time-dependent AUC(Uno's) 계산 시점(day, 콤마 구분). 기본 12/24/36개월"
             "(365,730,1095). 3/6/12/24개월처럼 다른 구간 분포를 보고 싶을 때 예: "
             "--auc-days 91,182,365,730. wandb의 auc_12m/24m/36m 로깅은 --auc-days가 기본값이 "
             "아니면 값이 없을 수 있어(.get, nan 처리) — 콘솔 로그(_log_line)에는 실제 계산된 "
             "모든 시점의 AUC가 항상 개별 표시된다.",
    )
    parser.add_argument(
        "--sam", action="store_true",
        help="2026-07-27: SAM(Sharpness-Aware Minimization, utils/sam.py)으로 AdamW를 감싼다 — "
             "현재 지점이 아니라 근방(반경 --sam-rho) 최악점 기준 gradient로 업데이트해 flat "
             "minimum을 명시적으로 찾는다. 초기화만 바꿔도 external이 크게 흔들리는 현상(local "
             "minimum 로또)에 대한 직접적 대응 가설 검증용. 배치당 forward+backward를 2번 하므로 "
             "학습 시간이 대략 2배가 된다.",
    )
    parser.add_argument(
        "--sam-rho", type=float, default=0.05,
        help="--sam의 perturbation 반경(기본 0.05, SAM 원논문 기본값). 클수록 더 넓은 flat 영역을 "
             "요구한다.",
    )
    parser.add_argument(
        "--sam-wsi-only", action="store_true",
        help="2026-08-06: --sam과 함께 사용 — perturbation을 WSI 브랜치 파라미터(model.cnn/vit/"
             "attn_pool/multi_pool/component_coattn/dispersion_scale, 존재하는 것만)에만 적용하고 "
             "나머지(RNA/clinical 인코더, risk_head, aux head)는 rho=0인 별도 param group으로 둬서 "
             "사실상 일반 AdamW로 학습한다(utils/sam.py의 SAM은 이미 param_group 단위로 rho를 "
             "따로 가질 수 있어 SAM 클래스 자체는 손댈 필요가 없었다). WSI 인코더(파라미터 수가 "
             "가장 많고 patient-level 라벨 대비 과적합 여지가 가장 큰 부분)만 flat minimum을 "
             "찾도록 강제하는 게, 이미 상대적으로 가벼운 RNA/clinical 브랜치까지 같이 흔드는 것보다 "
             "나은지 확인하는 ablation. --sam 없이 쓰면 무시된다.",
    )
    parser.add_argument(
        "--clinical-lr-mult", type=float, default=1.0,
        help="2026-08-15: clinical_encoder(combine-mode=concat)/clinical_linear(cox_add) "
             "param group에만 이 배율을 곱한 lr을 준다(1.0=끄기, --sam과 동시 사용 불가). "
             "scripts/diagnose_m2_branch_swap.py 실측 — 공동학습된 clinical 브랜치가 M5(clinical "
             "단독) 대비 internal -0.075/external -0.018 떨어지는데 WSI 브랜치는 M1 대비 거의 "
             "안 상한다 — 파라미터 수가 훨씬 적은 clinical이 WSI와 같은 lr로 경쟁하면 밀려난다는 "
             "가설을 검증. 켜면 model_prefix에 _CLR{배율}이 붙는다.",
    )
    parser.add_argument(
        "--rna-lr-mult", type=float, default=1.0,
        help="--clinical-lr-mult와 동일 관례, rna_encoder/rna_linear(M3/M4/PMA)에 적용. "
             "clinical-lr-mult가 fold1에서 확인된 효과가 RNA 브랜치에도 적용되는지 검증. "
             "켜면 model_prefix에 _RLR{배율}이 붙는다.",
    )
    parser.add_argument(
        "--wsi-lr-mult", type=float, default=1.0,
        help="2026-09-03: --clinical-lr-mult/--rna-lr-mult와 동일 관례, WSI 브랜치(cnn/vit/"
             "attn_pool/multi_pool/component_coattn/dispersion_scale, --sam-wsi-only와 동일 "
             "정의)에 적용. diagnose_m4_branch_gradients.py 실측 — RNA 인코더 gradient norm이 "
             "WSI보다 30 epoch 내내 1.6~1.9배 컸다(RNA가 리딩 팩터, 사용자 확인) — RNA를 누르는 "
             "대신 WSI/clinical 양쪽을 같이 끌어올려 경쟁력을 맞추는 시도. 켜면 model_prefix에 "
             "_WLR{배율}이 붙는다.",
    )
    parser.add_argument(
        "--lr-mult-warmup-epochs", type=int, default=0,
        help="2026-08-15: --clinical-lr-mult/--rna-lr-mult/--wsi-lr-mult 전용. 배율을 학습 시작부터 목표값(예: "
             "20배) 그대로 쓰는 대신, 이 epoch 수에 걸쳐 1.0배에서 목표 배율까지 선형으로 올린다. "
             "M4+RLR20 fold2/3 실측 — 20배를 처음부터 쓰면 rna_encoder가 1~2 epoch 만에 31명짜리 "
             "val set에 우연히 잘 맞는(하지만 불안정한) 지점으로 점프해버리고 그 뒤로 val이 단조 "
             "하락(과적합)하는 패턴이 확인됨(baseline은 val이 8~11 epoch에 걸쳐 서서히 정점에 "
             "도달). 0(기본)이면 기존과 동일(warmup 없이 목표 배율 그대로). 켜면 model_prefix에 "
             "_LRMW{epochs}가 붙는다.",
    )
    parser.add_argument(
        "--warm-start-clinical", type=str, default=None,
        help="2026-08-15: ClinicalOnly(--M5) 체크포인트 경로. 학습 시작 전 model.clinical_encoder를 "
             "이 체크포인트의 clinical_encoder 가중치로 초기화한다(0부터 WSI와 경쟁하며 학습하는 "
             "대신, 이미 clinical 단독으로 수렴한 지점에서 joint fine-tuning을 시작). "
             "clinical_encoder가 없는 모델(cox_add 등)에서 쓰면 에러.",
    )
    parser.add_argument(
        "--warm-start-rna", type=str, default=None,
        help="--warm-start-clinical과 동일 관례, RNAOnly(--M6) 체크포인트로 model.rna_encoder를 "
             "초기화한다.",
    )
    parser.add_argument(
        "--freeze-rna", action="store_true",
        help="2026-09-03: --warm-start-rna와 함께 사용 — rna_encoder를 M6 체크포인트로 초기화한 "
             "뒤 requires_grad=False로 고정해, RNA를 (fine-tuning 없이) 고정 특징 추출기처럼만 "
             "쓴다(사용자 제안: 'RNA branch를 백본처럼'). diagnose_m4_branch_gradients.py에서 "
             "RNA gradient norm이 WSI보다 30 epoch 내내 1.6~1.9배 컸던 것 — RNA를 아예 안 건드리게 "
             "고정하면 WSI/clinical이 RNA와 gradient 경쟁 없이 온전히 학습 신호를 받는지 검증. "
             "--warm-start-rna 없이 쓰면 에러(무작위 초기화 상태로 고정하는 건 의미가 없음).",
    )
    parser.add_argument(
        "--auto-branch-balance", action="store_true",
        help="2026-08-15: --clinical-lr-mult/--rna-lr-mult처럼 고정 배율을 손으로 정하는 대신, "
             "매 optimizer step 직전에 clinical_encoder/rna_encoder 브랜치의 gradient norm을 "
             "나머지(WSI 등) 파라미터의 gradient norm에 맞춰 실시간으로 재조정한다(스케일 배율을 "
             "[0.1, 100] 범위로 clamp). 학습 내내 배율이 고정되지 않고 각 스텝의 실제 gradient "
             "크기 차이에 맞춰 자동으로 움직인다는 점이 lr-mult와의 차이 — 둘을 동시에 켜도 된다.",
    )
    parser.add_argument(
        "--ogm-ge-alpha", type=float, default=None,
        help="2026-08-15: Peng et al. CVPR 2022(OGM-GE)를 우리 Cox 구조에 맞게 이식(train.py::"
             "train_one_epoch::_ogm_ge_modulate 참조). --combine-mode cox_add 전용(WSI/clinical "
             "항이 additive로 깨끗이 분리되는 유일한 지점). 이번 배치에서 WSI/clinical 단독 "
             "risk의 -Cox loss를 '판별력 점수'로 삼아, 더 앞서가는 쪽의 gradient를 "
             "tanh(alpha*|score차|) 계수로 억제하고 억제한 만큼 노이즈를 더한다(GE, epoch가 "
             "진행될수록 anneal). None(기본)이면 꺼짐. alpha가 클수록 억제가 강함 — 처음엔 "
             "1.0 근처로 시도.",
    )
    parser.add_argument(
        "--entropy-reg-weight", type=float, default=0.0,
        help="2026-08-31: attn_pool의 patch attention entropy(0~1, 1=완전균등)를 Cox loss에 "
             "직접 벌점으로 더해, 학습 중에 균등분포 붕괴를 억제해본다(train.py::train_one_epoch "
             "_compute_loss 참조). 배경 — findings_backlog.md: entropy 0.999+ 붕괴가 PAAD/BRCA, "
             "나이스트롬 유무와 무관하게 재현됨. 이미 학습된 체크포인트에 재학습 없이 temperature만 "
             "낮추는 후처리 sharpening은 오히려 성능을 떨어뜨렸다(2026-08-31 diagnose 결과 — T=1로 "
             "학습된 raw score를 T<1로 재해석하면 신호뿐 아니라 노이즈까지 같이 증폭됨) — 그래서 "
             "이번엔 처음부터 이 벌점을 알고 학습하게 한다. 0(기본)이면 꺼짐. attn_weights를 "
             "반환하는 모든 WSI 모델(M1/M2/M4/M4A/PMA/MCAT/PORPOISE 등)에서 공통 동작.",
    )
    parser.add_argument(
        "--swa", action="store_true",
        help="2026-07-27: SWA(Stochastic Weight Averaging) — 학습 후반부(--swa-start-frac 이후) "
             "매 epoch의 가중치를 평균 낸 별도 모델을 유지하고, 학습 종료 후 이 평균 모델도 "
             "internal/external test에 추가로 평가해 별도 리포트한다(기존 best-val/마지막-epoch "
             "리포트는 그대로 유지, SWA는 세 번째 관점을 더하는 것). torch.optim.swa_utils 사용.",
    )
    parser.add_argument(
        "--swa-start-frac", type=float, default=0.75,
        help="--swa 평균을 시작할 시점(전체 epoch 대비 비율, 기본 0.75 = 마지막 25%%만 평균).",
    )
    parser.add_argument(
        "--swad", action="store_true",
        help="2026-08-12: [Poor-man's SWAD, Cha et al. 2021] --swa는 고정 비율(마지막 N%%)을 "
             "무조건 평균 내지만, SWAD는 'val 성능이 최고점 근방(flat/plateau)에 머무는 구간'만 "
             "찾아서 평균한다 — model soup 파일럿에서 fold 하나가 시드 간 완전히 다른 basin에 "
             "떨어진 것(internal c=0.4963)을 본 뒤, 사후 평균이 아니라 학습 중 flat minimum을 "
             "능동적으로 좇는 게 나은지 확인하는 실험. 원 논문은 mini-batch 단위로 조밀하게 "
             "평균하지만 여기서는 매 epoch(val eval 있는 시점)이 가장 조밀한 단위라 epoch 단위로 "
             "근사한다 — best epoch를 중심으로 val c_index가 (best - --swad-tolerance) 이상인 "
             "연속 구간을 좌우로 넓혀가며 그 구간의 epoch 가중치만 균등 평균. --swa와 동시 사용 가능"
             "(서로 독립적으로 추가 평가 리포트를 남김).",
    )
    parser.add_argument(
        "--swad-tolerance", type=float, default=0.02,
        help="--swad plateau 구간을 정할 val c_index 허용 오차(기본 0.02 = best epoch 대비 "
             "0.02 이내인 연속 epoch까지 포함).",
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=None,
        help="2026-08-12: 주어지면, best-val c_index가 이 patience(epoch 수)만큼 연속으로 "
             "갱신되지 않으면 학습을 조기 종료한다(--epochs를 크게 잡아두고 실제로 필요한 "
             "만큼만 도는 용도, 예: uni2official처럼 patch 수가 훨씬 많아 수렴이 느린 backbone). "
             "best-val checkpoint는 기존과 동일하게 그 시점 그대로 저장돼 있으므로 최종 성능"
             "(internal/external 평가)에는 영향이 없다 — GPU 시간만 아낀다. 기본(None)이면 "
             "비활성화(기존 동작 그대로 --epochs 끝까지 학습).",
    )
    parser.add_argument(
        "--no-patient-shuffle", action="store_true",
        help="2026-07-27: train DataLoader의 shuffle=True(기본, 매 epoch 환자 처리 순서를 다시 "
             "섞음)를 끄고 항상 고정 순서로 순회한다. patch 서브샘플링 패턴을 격리해도 --full-train "
             "시드 간 분산(std~0.02)이 그대로 남은 뒤, 남은 후보(모델 초기화 vs 환자 처리 순서) 중 "
             "순서 쪽을 분리 검증하기 위한 용도.",
    )
    parser.add_argument(
        "--rna-aux-weight", type=float, default=0.0,
        help="WSI 표현이 RNA 발현도 예측하도록 하는 보조과제(auxiliary task) 가중치, "
             "models/rna_predictor.py::RNAPredictionHead. 0.0(기본)이면 비활성. RNA를 쓰는 "
             "모델(--M4/--M4A/--M4B/--PM4/--PMA)에서만 적용되며(rna_encoder 필요), attn_pool의 "
             "RNA 개입과 무관하게 ViT 직후 mean-pooled 표현(RNA-free)에서 예측한다 - HE2RNA류 "
             "설계. cox loss에 이 가중치를 곱해 더한다. 0.0 초과면 wandb/checkpoint에 _AUX "
             "접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--rna-snn", action="store_true",
        help="2026-08-13: models/rna_encoder.py::RNAEncoder를 PORPOISE(Chen et al. 2022) "
             "SNN_Block 스타일(Linear->ELU->AlphaDropout->Linear, LayerNorm 제거)로 교체한다 "
             "— 기본(GELU+LayerNorm+Dropout) 대비 표준 Dropout보다 덜 파괴적인 AlphaDropout이 "
             "RNA 브랜치의 반복 관측된 과적합(train-val c_index 격차)을 줄이는지 보는 파일럿. "
             "현재 --M6(RNAOnly)에만 배선됨.",
    )
    parser.add_argument(
        "--modality-dropout-p", type=float, default=0.0,
        help="RNA를 쓰는 모델(--M4/--M4A/--M4B/--PM4/--PMA, rna_encoder 필요)에서, 학습 중 "
             "이 확률로 z_rna를 통째로 0벡터로 지운다(train.py::_patient_risk). "
             "diagnose_wsi_gradients.py 진단(RNA 인코더 gradient norm이 WSI의 ~4배)에서 드러난 "
             "modality imbalance — risk_head가 강한 RNA 신호에 안주해 WSI/clinical 브랜치가 "
             "undertrained되는 문제 — 에 대한 대응(일반 멀티모달 학습 문헌의 modality dropout, "
             "예: Peng et al. CVPR 2022 OGM-GE가 다루는 것과 같은 계열의 문제를 가장 단순한 "
             "방식으로 완화). --rna-aux-weight 보조 loss는 항상 진짜 RNA 값을 타깃으로 쓰므로 "
             "영향받지 않는다 — 지워지는 건 메인 risk 경로의 z_rna뿐. 0.0(기본)이면 비활성. "
             "0.0 초과면 model_prefix에 _MODDROP{value}가 붙는다.",
    )
    parser.add_argument(
        "--clinical-staging", action="store_true",
        help="ClinicalEncoder 입력에 age/sex 뿐 아니라 AJCC 병기(T/N/M)+grade도 추가한다 "
             "(models/clinical_encoder.py::ClinicalEncoder(use_staging=True)). data/clinical_"
             "{tcga,cptac}.csv를 쓰는 모델(--M2/--M4/--M4A/--M4B/--PM4/--PMA/--M4A_FF/--M2_FF/--M5)"
             "에서만 사용 가능. 기본은 미사용(age/sex만) - 'age/sex만 쓰라'는 기존 지시가 있어 "
             "두 버전(있음/없음)을 다 비교할 수 있게 별도 플래그로 뒀다. 켜면 wandb/checkpoint에 "
             "_STG 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--clinical-margin", action="store_true",
        help="train_light.py --clinical-margin과 동일 — ClinicalEncoder 입력에 절제연 상태"
             "(residual_disease: R0=완전절제 < R1 < R2=잔존종양)를 추가한다. --clinical-staging"
             "(T/N/M/grade)과 별개 플래그 — 함께 켤 수도 있다. --M2/--M4/--M4A/--M4B/--PM4/"
             "--PMA/--M4A_FF/--M2_FF/--M5에서 사용 가능. 켜면 wandb/checkpoint에 _R 접미사가 "
             "자동으로 붙는다.",
    )
    parser.add_argument(
        "--no-age-sex", action="store_true",
        help="train_light.py --no-age-sex와 동일 — --clinical-margin과 함께 사용, age/sex를 빼고 "
             "margin(/staging)만 입력으로 쓴다. 켜면 model_prefix에 _ONLY가 추가로 붙는다.",
    )
    parser.add_argument(
        "--clinical-mutation", action="store_true",
        help="train_light.py --clinical-mutation과 동일 — ClinicalEncoder 입력에 PDAC 4대 driver "
             "gene mutation status(KRAS/TP53/SMAD4/CDKN2A, data/clinical_{tcga,cptac}.csv의 "
             "{gene}_mut 컬럼, models/clinical_encoder.py::MUTATION_FIELDS)를 추가한다. "
             "--clinical-staging/--clinical-margin과 별개 플래그 — 함께 켤 수 있다. 2026-09-03 "
             "기준 --M4에서만 지원(models/vit_m4.py::ViT_M4). 켜면 wandb/checkpoint에 _MUT "
             "접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--use-cnv", action="store_true",
        help="train_light.py --use-cnv와 동일 — data/extract_cnv.py 산출물(pathway8 163유전자 "
             "범위 copy number, log2-ratio+z-score, 카테고리 8개 평균)을 RNA 브랜치 뒤에 "
             "concat한다(data/dataset.py::WSISurvivalDataset(with_cnv=True), --rna-genes 종류와 "
             "무관하게 항상 +8차원). --M4/--M6/--M6X 등 RNA를 쓰는 모델에서 사용 가능. 켜면 "
             "wandb/checkpoint에 _CNV 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--drop-component", type=str, default=None, choices=["mean", "std", "attn", "top"],
        help="--PMA 전용(models/multi_component_pooling.py::MultiComponentPooling). "
             "2026-08-09: scripts/diagnose_pma_component_reliance.py의 사후 zero-ablation에서 "
             "4개 관점(mean/std/attn/top-k) 중 어느 것을 지워도 손해가 아니었던(top-k 제거가 "
             "external에 소폭 긍정적) 결과를 받아, 구조적으로 하나를 아예 빼고 처음부터 "
             "재학습해 internal이 실제로 오르는지 검증한다(파라미터/입력 차원이 줄어든 만큼 "
             "과적합 압력이 줄 수 있다는 가설). co-attention은 토큰 개수에 무관하게 동작해 "
             "risk_head 등 다른 차원엔 영향 없음. 기본 None(4개 다 사용, 기존 동작). 켜면 "
             "model_prefix에 _NO{COMPONENT}가 붙는다(예: --drop-component top -> _NOTOP).",
    )
    parser.add_argument(
        "--rna-combine-mode", type=str, default="concat", choices=["concat", "cox_add"],
        help="--PMA 전용(models/vit_pma.py::ViT_PMA). 2026-08-09: clinical의 cox_add 원리를 "
             "RNA에도 이식 — RNA는 여전히 component_coattn의 query로 WSI pooling을 guide하지만"
             "(이 경로는 그대로 둠), risk_head에 z_rna를 직결 concat하던 경로만 떼어내 별도의 "
             "고전적 Cox 가산항(rna_linear, zero-init)으로 바꾼다. --rna-gate-only(z_rna의 "
             "기여 자체를 차단)와 달리 z_rna의 marginal 기여를 구조적으로 분리해서 보존한다는 "
             "점이 다르다. 기본 concat(기존 동작). 켜면 model_prefix에 _RNACOXADD가 붙는다.",
    )
    parser.add_argument(
        "--skip-patch-vit", action="store_true",
        help="--PMA 등 ViT_M1 계열 공통(models/vit_m1.py). 패치 간 self-attention을 섞는 "
             "self.vit(ViTEncoder, Nystrom/full attention 1-layer)을 아예 생성하지 않고, "
             "backbone(UNI2 등) 출력을 embed_dim으로 projection한 patch_tokens를 그대로 "
             "attn_pool(MultiComponentPooling 등)에 넘긴다. 2026-08-11: UNI2-h처럼 이미 강한 "
             "사전학습 backbone을 쓸 때, 적은 표본(환자 ~150명)으로 처음부터 학습하는 이 작은 "
             "patch-mixing transformer가 오히려 좋은 patch token을 흐릴 수 있다는 가설의 구조적 "
             "ablation. 켜면 model_prefix에 _NOVIT가 붙는다.",
    )
    parser.add_argument(
        "--coord-embed", action="store_true",
        help="ViT_M1 계열 공통(models/vit_m1.py::ViT_M1). 학습 파라미터 없는 sinusoidal "
             "위치 인코딩(models/vit_encoder.py::SpatialPositionEmbedding, --skip-patch-vit가 "
             "꺼져있을 때 self.vit 내부에서 이미 쓰던 것과 동일)을 patch_tokens에 잔차로 "
             "더한 뒤 attn_pool에 넘긴다. --skip-patch-vit(NOVIT)와 같이 쓰면 그 경우 "
             "완전히 사라졌던 패치 위치 정보를 attn_pool의 gate(attn_v/attn_u)에 되살려주는 "
             "게 목적 — attn-dispersion은 attn_weights에 대한 post-hoc 페널티일 뿐 토큰 "
             "자체엔 안 섞이는 것과 다르다. 켜면 model_prefix에 _COORD가 붙는다.",
    )
    parser.add_argument(
        "--coord-embed-concat", action="store_true",
        help="--coord-embed 전용 세부옵션. 기본(잔차 add) 대신 [patch_tokens ‖ coord_embed] -> "
             "Linear->LayerNorm->GELU 융합층으로 다시 embed_dim에 투영한다 — 위치 정보를 "
             "별도 채널로 유지해 fusion 레이어가 얼마나 반영할지 직접 학습하게 한다. 켜면 "
             "model_prefix에 _CAT이 붙는다.",
    )
    parser.add_argument(
        "--coord-embed-learnable-scale", action="store_true",
        help="--coord-embed 전용 세부옵션(--coord-embed-concat과는 배타적, concat이면 무시됨). "
             "dispersion_scale과 동일 관례로 잔차 add 전에 학습되는 스칼라 배율(0.2 초기화)을 "
             "곱한다. 켜면 model_prefix에 _SC가 붙는다.",
    )
    parser.add_argument(
        "--coord-embed-shuffle", action="store_true",
        help="--coord-embed 전용 대조군. forward마다 patch 순서를 무작위로 섞은 coords를 "
             "coord_embed에 넣어(슬라이드 전체 좌표 분포는 그대로, patch-position 대응만 "
             "파괴) coord-embed-concat의 개선이 진짜 위치 정보 때문인지 coord_fusion 레이어 "
             "자체의 capacity/정규화 효과인지 구분한다. 켜면 model_prefix에 _SHUF가 붙는다.",
    )
    parser.add_argument(
        "--wsi-extra-mlp", action="store_true",
        help="--coord-embed와 무관, 독립 플래그(models/vit_m1.py::ViT_M1). "
             "coord-embed-concat의 이득이 좌표(--coord-embed-shuffle로 확인)와 무관하게 "
             "coord_fusion이라는 추가 비선형 레이어 자체에서 왔다는 결론을 가장 순수한 "
             "형태로 재검증 — 좌표 인코딩을 아예 계산하지 않고 patch_tokens에 곧바로 "
             "Linear(D->D)->LayerNorm->GELU 레이어 하나만 추가로 통과시킨다. 켜면 "
             "model_prefix에 _XMLP가 붙는다.",
    )
    parser.add_argument(
        "--top-frac", type=float, default=0.1,
        help="--PMA 전용(models/multi_component_pooling.py::MultiComponentPooling). top-k-mean "
             "성분이 attention 상위 몇 %%의 패치를 평균할지(기본 0.1=10%%). 2026-08-09: "
             "--drop-component top(top-k를 아예 제거)이 internal을 올린 것을 보고, 'top-k 자체가 "
             "쓸모없다'가 아니라 '상위 10%%가 표본이 너무 작아 노이즈에 민감했다'는 가설을 "
             "검증하기 위해 노출 — 0.25 등으로 키우면 같은 attention-상위 컨셉을 유지하면서 "
             "표본을 넓힐 수 있다. 기본값(0.1)과 다르면 model_prefix에 _TOPFRAC{value}가 붙는다.",
    )
    parser.add_argument(
        "--pooling-mode", type=str, default="coattn", choices=["coattn", "selfattn"],
        help="--M2_POOL(models/vit_m2_pool.py::ViT_M2_Pool) 전용. 4개 pooling 관점(mean/std/"
             "attn/top-k)을 합치는 방식 — 'coattn'(기본)은 z_clinical(age/sex)을 co-attention "
             "query로 씀. 'selfattn'은 models/vit_m1_pool.py::SelfAttentionPooling을 재사용해 "
             "clinical 개입 없이 4개 관점이 서로 self-attention한다. 2026-08-07: coattn 버전이 "
             "UNI+age/sex만으로 external 0.49~0.51(랜덤 수준, M1_POOL의 self-attention 단독 "
             "0.556보다도 낮음)에 그쳐 — age/sex 신호가 약해 co-attention query로 쓰면 잘못된 "
             "기준으로 관점을 고르는 것으로 추정, selfattn+--combine-mode cox_add 조합으로 "
             "재검증한다. 켜면 model_prefix에 _SELFATTN 접미사가 붙는다.",
    )
    parser.add_argument(
        "--combine-mode", type=str, default="concat", choices=["concat", "cox_add"],
        help="--PMA(models/vit_pma.py::ViT_PMA)와 --M2(models/vit_m2.py::ViT_M2)에서 지원. "
             "train_light.py --M7 --combine-mode의 cox_add를 이식 — clinical(age/sex[/margin, "
             "PMA만]/staging)을 z_clinical 임베딩으로 concat하지 않고, risk_head(WSI(+RNA) 임베딩)에 "
             "고전적 Cox 가산항(clinical_linear, 파라미터 raw_dim개, zero-init이라 학습 시작 "
             "시점엔 clinical 없는 모델과 동일)으로 직접 더한다. "
             "PMA_INT1500_SS_AUX_R_DISP(no-aug, concat)가 external에서 M6/M7보다도 나은 걸 보고, "
             "M7에서 cox_add가 R_ONLY(margin만)의 internal을 크게 끌어올렸던 효과가 PMA에도 "
             "재현되는 걸 확인했다(2026-08-06) — M2(ABMIL)에서도 재현되는지 확인하는 ablation "
             "(2026-08-07). 켜면 model_prefix에 _COX_ADD 접미사가 붙는다.",
    )
    parser.add_argument(
        "--stage-aux-weight", type=float, default=0.0,
        help="WSI 표현이 T-stage/grade도 예측하도록 하는 보조과제(auxiliary task) 가중치, "
             "models/stage_predictor.py::StagePredictionHead. --rna-aux-weight와 동일한 설계 "
             "(RNA-free/clinical-free mean-pooled 표현에서 예측, 예측값은 버리고 그래디언트만 "
             "WSI 인코더 정규화에 쓴다). N/M-stage는 원발암 WSI만으로 판단 근거가 없어 T-stage/"
             "grade만 타깃으로 한다. WSI를 쓰는 모델(--M1 등, hasattr(model,'cnn'))에서만 적용 "
             "가능. 0.0(기본)이면 비활성, 0.0 초과면 wandb/checkpoint에 _AUX2 접미사가 자동으로 "
             "붙는다(--rna-aux-weight의 _AUX와 구분).",
    )
    parser.add_argument(
        "--external", action="store_true",
        help="internal test(같은 코호트 held-out)와 별도로, 학습에 전혀 쓰지 않은 반대 코호트 "
             "전체(tcga↔cptac 자동 선택)를 external test로 평가한다. 기본은 미사용(off) — "
             "켜려면 --external을 지정한다. --dataset both는 반대 코호트가 없어 함께 쓰면 에러.",
    )
    parser.add_argument(
        "--exclude-case-ids", type=str, default=None,
        help="2026-07-26: 콤마로 구분한 case_id 목록을 train set에서만 제외한다(val/test/external은 "
             "그대로 유지 — split 자체를 다시 계산하지 않고, 이미 만들어진 train_ds.items에서 "
             "해당 환자 행만 걸러낸다). seed84 train split에만 있는 '슬라이드 1장짜리 환자' 2명"
             "(TCGA-IB-7645, TCGA-YH-A8SY)이 external 부진의 원인인지 검증하기 위한 용도.",
    )
    parser.add_argument(
        "--full-train", action="store_true",
        help="2026-07-26: 6:2:2 stratified split 없이 --dataset 코호트 전체를 train으로 쓴다 "
             "(val/internal test 자체가 존재하지 않음). seed별 91명 train 분할의 '표본 운'이 "
             "external 성능 차이를 만든다는 관찰(seed42 vs 84/126) 이후, 가용 데이터를 전부 "
             "학습에 쓰면 external이 어떻게 되는지 보기 위한 실험용 플래그. val이 없어 best-val "
             "체크포인트 선택이 불가능하므로 마지막 epoch 모델로 external을 정확히 1회 평가한다 "
             "(--external과 함께 써야 의미가 있다).",
    )
    parser.add_argument(
        "--image", action="store_true",
        help="패치 jpg/png를 매 forward마다 ResNet50으로 직접 인코딩 (기본: data/extract_features.py로 "
             "사전 추출한 features.pt 사용)",
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet50",
        choices=["resnet50", "uni", "uni2", "resnet50_norm", "uni2official", "uni2native"],
        help="frozen tile encoder 선택 (기본: resnet50=Lunit SwAV, 2048-dim). uni는 UNI ViT-L/16"
             "(1024-dim, 224 리사이즈) — 미리 `python -m utils.extract_features --backbone uni`로 "
             "features_uni.pt를 뽑아둬야 한다(HuggingFace gated repo 접근 승인 + .env HF_TOKEN 필요). "
             "uni2는 UNI2-h ViT-H/14(models/uni2_encoder.py, 1536-dim, 512 리사이즈) — HPC에서 "
             "`python -m utils.extract_features --backbone uni2`(sbatch/extract_features_uni2_array_hpc.sh)"
             "로 뽑은 features_uni2.pt를 <patches_root>/<slide_id>/ 아래 그대로 두면 된다(uni와 "
             "별도 gated repo 승인 필요, MahmoodLab/UNI2-h). "
             "resnet50_norm은 Macenko stain-normalized 후 같은 ResNet50/Lunit SwAV로 재추출한 "
             "feature(features_norm.pt, utils/extract_features_stain_norm.py) — 인코더 자체는 "
             "resnet50과 동일(2048-dim), 캐싱 파일만 다르다. "
             "uni2official은 MahmoodLab이 공식 스펙(256px@20x, ~0.5MPP)으로 직접 뽑아 배포한 "
             "UNI2-h feature(HuggingFace dataset MahmoodLab/UNI2-h-features, gated) — 인코더는 "
             "uni2와 동일(1536-dim)하지만 patch grid가 우리 자체 추출본과 달라 coords도 별도 "
             "파일에서 읽는다(scripts/convert_uni2h_official_features.py 산출물 필요). "
             "uni2native는 uni2official과 같은 스펙(256px@20x)을 우리 raw WSI에서 우리 파이프라인"
             "(data/preprocess.py --target-mpp 0.5 --tile-size 256)으로 직접 재타일링한 버전 — "
             "uni2official이 가진 두 confound(DX 슬라이드만 포함, coords가 level0 픽셀 단위라 "
             "attn_dispersion 스케일이 ~4000배 어긋남)를 피한다(scripts/reconcile_uni2native_features.py "
             "산출물 필요).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="cfg.data.num_workers 덮어쓰기(기본: config.py 값 그대로, 0). DataLoader.__getitem__은 "
             "patch_paths/메타데이터만 반환하는 가벼운 작업이라(실제 이미지 디코딩+증강은 여기서 "
             "안 함, --tile-decode-workers 참조) precomputed=False에서도 효과가 제한적일 수 있다.",
    )
    parser.add_argument(
        "--tile-decode-workers", type=int, default=None,
        help="cfg.model.tile_decode_workers 덮어쓰기(기본: config.py 값 그대로, 4). --tile-augment"
             "(--image, CPU에서 RandomFlip/ColorJitter/GaussianBlur를 매 forward 실시간 적용)의 "
             "실제 병목 — models/vit_m1.py::_patch_tokens가 이 개수만큼 스레드로 타일 디코딩+증강을 "
             "미리 돌려 GPU 연산과 겹친다(2026-07-22 도입, 그동안 4로 하드코딩되어 있었음). SLURM "
             "--cpus-per-task로 예약한 CPU 개수만큼(예: 8) 줘야 그 CPU를 실제로 다 쓴다.",
    )
    parser.add_argument(
        "--cache-val-tiles", action="store_true",
        help="--tile-augment --image와 함께, val split도 train처럼 RAM에 프리로드한다(기본 꺼짐). "
             "2026-08-04: evaluate()가 train_eval/val 둘 다 tile_cache 없이 매 epoch 디스크에서 "
             "새로 디코딩하고 있던 게 느린 파일시스템(HPC 등)에서 epoch 1도 못 넘기는 병목이었다 — "
             "train_eval은 train_ds의 tile_cache를 그냥 재사용(추가 메모리 0)하도록 항상 고쳤고, "
             "val은 별도 캐시가 필요해(추가 RAM 필요, val 규모만큼) 옵트인으로 뒀다. 로컬처럼 RAM이 "
             "빠듯한 머신(train 캐시만으로 32GB 중 ~22GB를 이미 씀, findings_backlog.md 스와핑 사고 "
             "전례)에서는 끄고, HPC처럼 RAM 여유가 큰 곳에서만 켜라.",
    )
    parser.add_argument(
        "--cache-external-tiles", action="store_true",
        help="--tile-augment --image와 함께, external(반대 코호트 전체) split도 RAM에 프리로드한다"
             "(기본 꺼짐) — --cache-val-tiles와 동일한 이유. external은 보통 한 run당 1회만 평가돼 "
             "val만큼 반복 이득은 없지만(매 epoch 아님), 그 1회 자체가 코호트 전체(TCGA/CPTAC 상대편) "
             "라 val보다 크고 디스크에서 새로 읽으면 여전히 느리다. 로컬 실측 기준 cptac 전체 "
             "~28,000타일(~21GB), tcga 전체는 더 크다 — 128GB급 HPC에서만 켜라.",
    )
    parser.add_argument(
        "--patches-root-tcga", type=str, default=None,
        help="cfg.data.patches_root_tcga 덮어쓰기(기본: config.py 값 그대로, data/patches_tcga). "
             "재타일링된 패치(예: data/patches_tcga_512)로 학습/평가할 때 사용.",
    )
    parser.add_argument(
        "--patches-root-cptac", type=str, default=None,
        help="cfg.data.patches_root_cptac 덮어쓰기(기본: config.py 값 그대로, data/patches_cptac). "
             "재타일링된 패치(예: data/patches_cptac_512)로 학습/평가할 때 사용.",
    )
    # [LateFusion] --fusion 플래그로 LateFusionViT 사용 여부 선택
    # 미지정 시 기존 ViT_M1(ViT+ABMIL)로 동작 — ablation baseline 유지
    parser.add_argument(
        "--full-attention", action="store_true",
        help="ViT/Nystromformer 블록의 self-attention을 Nystrom 근사 대신 표준 O(N^2) "
             "attention(nn.MultiheadAttention)으로 교체한다(cfg.model.use_nystrom=False). "
             "슬라이드당 패치 수 실측(평균 131/중앙값 67/최대 544)이 num_landmarks(128)보다도 "
             "작은 경우가 절반 이상이라, Nystrom 근사가 오히려 패딩 토큰을 landmark에 섞어 "
             "역효과를 냈을 수 있다는 의심 검증용(findings_backlog.md). 이 규모면 O(N^2)도 "
             "GPU에 부담 없다. 켜면 wandb/checkpoint에 _FULLATTN 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--no-spatial-embed", action="store_true",
        help="ViT/Nystromformer 입력에서 좌표 기반 SpatialPositionEmbedding을 아예 뺀다"
             "(patch_tokens를 그대로 사용, cfg.model.use_spatial_embed=False). attention이 이미 "
             "uniform으로 붕괴해 있고(diagnose_wsi_reliance.py) 나이스트롬/full-attention 둘 다 "
             "landmark·정밀도 실험에서 뚜렷한 신호가 없었던 상황에서, 좌표 임베딩 자체가 최종 "
             "예측에 기여하는지 직접 확인하는 ablation(findings_backlog.md). 켜면 wandb/"
             "checkpoint에 _NOSPATIAL 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--rel-bias-attention", action="store_true",
        help="ViT/Nystromformer 블록의 절대좌표 SpatialPositionEmbedding을 빼고, 그 자리에 "
             "상대offset(Δrow,Δcol) 기반 attention bias(Swin류, models/vit_encoder.py::"
             "RelativeBiasFullAttention)를 넣은 전체(O(N^2)) self-attention으로 교체한다"
             "(cfg.model.use_rel_bias_attn=True, use_nystrom/use_spatial_embed는 자동으로 "
             "False). 2026-07-23: 얼린 Stage1 + 잔차 branch로 공간정보를 late-fusion하는 "
             "ResTopoMIL류 설계(models/spatial_residual.py)가 PAAD·BRCA 둘 다에서 실패한 뒤, "
             "WSI branch 안에서 RNA/Clinical과 함께 처음부터 end-to-end로 학습시키는 대안. "
             "켜면 wandb/checkpoint에 _RELBIAS 접미사가 자동으로 붙는다. --full-attention/"
             "--no-spatial-embed와 동시 사용 불필요(이미 포함됨).",
    )
    parser.add_argument(
        "--knn-bias-attention", action="store_true",
        help="--rel-bias-attention의 희소(sparse) 버전 — 전체(O(N^2)) attention 대신 각 패치의 "
             "kNN 이웃 k개(--knn-k, 기본 8)에만 attention한다(models/vit_encoder.py::"
             "KNNBiasAttention). 2026-07-23: BRCA(슬라이드당 패치 수 중앙값 10,309/최대 67,268)에 "
             "--rel-bias-attention을 그대로 돌리면 attention logit/bias 텐서가 N^2으로 터져 즉시 "
             "CUDA OOM — PAAD(N<=544) 전용이던 'O(N^2) 무부담' 가정이 BRCA에는 적용되지 않아 "
             "추가한 대안. cfg.model.use_knn_bias_attn=True, use_nystrom/use_spatial_embed는 "
             "자동으로 False. --rel-bias-attention과 동시 사용 시 dense가 우선한다. 켜면 wandb/"
             "checkpoint에 _KNNATTN 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--knn-k", type=int, default=8,
        help="--knn-bias-attention/--hybrid-attention 사용 시 패치당 kNN 이웃 수"
             "(cfg.model.knn_attn_k, 기본 8).",
    )
    parser.add_argument(
        "--hybrid-attention", action="store_true",
        help="같은 레이어에서 local(kNN-bias-attention)과 global(기존 Nystrom)을 병렬로 계산해 "
             "더한다(models/vit_encoder.py::HybridLocalGlobalAttention, cfg.model.use_hybrid_attn"
             "=True). --rel-bias-attention/--knn-bias-attention과 달리 use_nystrom/"
             "use_spatial_embed를 강제로 끄지 않는다 — global 경로가 계속 절대좌표 임베딩을 "
             "쓰기 때문. 2026-07-23: kNN 단독이 PAAD(pre-augment)에서 기존 baseline보다 낮게 "
             "나온 뒤(internal 0.6309->0.6094, external 0.6289->0.5880) 시도하는 대안. 켜면 "
             "wandb/checkpoint에 _HYBRIDATTN 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--spatial-autocorr", action="store_true",
        help="학습 파라미터 없는 공간 자기상관 특징(models/spatial_features.py::spatial_autocorr, "
             "Moran's I류 — 패치 임베딩과 kNN 이웃의 코사인 유사도 평균/표준편차 2개)을 risk_head "
             "5번째 관점으로 추가한다(cfg.model.use_spatial_autocorr=True, ViT_PMA 전용). "
             "2026-07-23: 학습형 spatial attention(kNN/hybrid)이 전부 baseline을 못 넘은 뒤 "
             "'새 attention 파라미터 자체가 과적합 유인'이라는 가설을 검증하는 저비용 대안. "
             "켜면 wandb/checkpoint에 _AUTOCORR 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--attn-dispersion", action="store_true",
        help="학습 파라미터 없는 attention 공간 분산 특징(models/spatial_features.py::"
             "attention_dispersion — attn_weights로 가중한 좌표의 표준편차 1개)을 risk_head "
             "5번째 관점으로 추가한다(cfg.model.use_attn_dispersion=True, ViT_PMA 전용). "
             "--spatial-autocorr와 독립적으로 켤 수 있다(순차 검증용). 켜면 wandb/checkpoint에 "
             "_DISP 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--tumor-type-embed", action="store_true",
        help="2026-08-13: 슬라이드가 종양/정상/미상(data/dataset.py::_slide_tumor_status, "
             "0/1/2)인지를 학습 가능한 임베딩으로 인코딩해 self.vit 입력 직전 patch_tokens에 "
             "더한다(models/vit_encoder.py — pos_embedding과 동일한 자리, 동일한 가산 패턴). "
             "findings_backlog.md 14번 항목(대표 슬라이드 1장으로 줄이면 오히려 악화)과 "
             "uni2official 대조실험(DX 슬라이드만 있으면 성능 하락)에서, 지금은 암묵적으로만 "
             "활용되는 슬라이드 타입 정보를 모델에 명시적으로 알려주면 더 잘 쓸 수 있는지 "
             "확인하는 파일럿. 현재 --PMA에만 배선됨. 켜면 wandb/checkpoint에 _TTE 접미사가 "
             "자동으로 붙는다.",
    )
    parser.add_argument(
        "--knn-fixed-bias-attention", action="store_true",
        help="--knn-bias-attention의 학습되는 RelativePositionBias(MLP)를 고정(학습 파라미터 없는) "
             "거리감쇠 커널 bias=-dist/tau로 교체한다(models/vit_encoder.py::KNNFixedBiasAttention, "
             "cfg.model.use_knn_fixed_bias_attn=True, use_nystrom/use_spatial_embed 자동 False). "
             "q/k/v/out projection은 그대로 학습되므로, '새 attention 파라미터 전체'가 아니라 "
             "'bias만 학습되는 것'이 과적합 유인인지 분리 검증하는 idea #3(2026-07-23). 켜면 "
             "wandb/checkpoint에 _FIXEDBIAS 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--bias-tau", type=float, default=50.0,
        help="--knn-fixed-bias-attention 사용 시 거리감쇠 스케일(cfg.model.knn_bias_tau, 기본 50). "
             "bias=-dist/tau — tau가 클수록 거리에 덜 민감(bias가 완만해짐). --learnable-tau가 "
             "켜지면 이 값은 초기값으로만 쓰인다.",
    )
    parser.add_argument(
        "--learnable-tau", action="store_true",
        help="--knn-fixed-bias-attention과 함께 사용 — tau를 고정 상수 대신 head별 학습 스칼라로 "
             "바꾼다(cfg.model.knn_bias_learnable_tau=True). PSA-MIL(WACV 2026, arXiv:2503.16284)의 "
             "'learnable distance-decayed prior'를 posterior=likelihood×prior의 log공간 덧셈 "
             "형태(이미 이 프로젝트의 'attention logit + bias' 구조와 동일)로 경량 재현한 버전 — "
             "head 개수(보통 2)만큼만 새로 학습되고 RelativePositionBias MLP(82개)보다 훨씬 작다. "
             "켜면 wandb/checkpoint에 _LEARNTAU 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--fix-nystrom-landmarks", action="store_true",
        help="2026-09-05: nystrom_attention 라이브러리의 알려진 결함(findings_backlog.md, "
             "config.py 주석) — 패치 수 n < num_landmarks(기본 128)면 라이브러리가 (128-n)개 "
             "zero 토큰을 F.pad로 채워 landmark에 섞어 넣는다(슬라이드 절반 이상이 이 경우, "
             "실측 중앙값 67). 라이브러리 코드는 안 건드리고, 매 forward 직전 "
             "self.attn.num_landmarks를 min(설정값, 실제 패치 수)로 clamp해서 우회한다"
             "(cfg.model.fix_nystrom_landmarks=True, models/vit_encoder.py::NystromEncoderLayer)."
             " --use-nystrom(기본값)에서만 의미 있음 — rel-bias/knn 계열 attention에는 영향 없음. "
             "켜면 wandb/checkpoint에 _NYSTROMFIX 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--knn-mean-agg", action="store_true",
        help="2026-09-05: '패치끼리 정보를 교환하되 attention/O(N^2) 없이' — kNN 이웃 k개(--knn-k)를 "
             "attention 가중치 없이 단순 평균(GraphSAGE mean-aggregator)해 자기 자신과 concat 후 "
             "선형변환(models/vit_encoder.py::KNNMeanAggregation, cfg.model.use_knn_mean_agg=True, "
             "use_nystrom/use_spatial_embed 자동 False). --rel-bias-attention이 A30(24GB)에서도 "
             "OOM난 뒤 시도하는 저메모리 대안 — attention logit/softmax 자체가 없어 메모리가 "
             "O(엣지 수)로 dense(O(N^2))보다 훨씬 가볍다. 켜면 wandb/checkpoint에 _KNNMEANAGG "
             "접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--cluster-attn", action="store_true",
        help="2026-09-05: '패치끼리 정보를 교환하되 O(N^2) 없이' 두 번째 대안 — 패치 N개를 "
             "K개(--n-clusters, 기본 16) 학습 가능한 클러스터 프로토타입에 소프트 배정해 "
             "슈퍼토큰으로 압축한 뒤, 그 K개끼리만 dense full self-attention(K가 작아 O(K^2)이 "
             "사실상 공짜)을 돌리고 같은 배정 가중치로 결과를 다시 각 패치에 broadcast한다"
             "(models/vit_encoder.py::HierarchicalClusterAttention, cfg.model.use_cluster_attn=True, "
             "use_nystrom/use_spatial_embed 자동 False). Slot Attention/Perceiver의 병목(bottleneck) "
             "attention과 같은 발상 — 사전학습된 클러스터(K-means)를 재사용하지 않고 end-to-end로 "
             "학습시킨다. 켜면 wandb/checkpoint에 _CLUSTERATTN{n_clusters} 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=16,
        help="--cluster-attn 사용 시 슈퍼토큰(클러스터 프로토타입) 개수(cfg.model.n_clusters, 기본 16).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="cfg.train.epochs(기본 30) 덮어쓰기 — 짧은 파일럿 실험용(예: 실시간 augmentation "
             "시간 확인, --epochs 10).",
    )
    parser.add_argument(
        "--dropout", type=float, default=None,
        help="cfg.model.dropout(기본 0.3) 덮어쓰기 — ViT/Nystromformer, ABMIL, RNA/Clinical "
             "인코더 전체가 공유하는 dropout rate 스윕용(findings_backlog.md 13번 항목 후속, "
             "risk head 자체 dropout과는 별개 실험). 기본값(None)과 다르면 wandb/checkpoint에 "
             "_DROP{dropout} 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=None,
        help="cfg.model.embed_dim(기본 64) 덮어쓰기 — WSI 브랜치 전체(CNN proj/ViT/pooling)의 "
             "폭. --rna-dim/--clinical-dim과 달리 이 값은 WSI 쪽 자체를 줄이는 용도(2026-07-28, "
             "PMA에서 clinical/RNA만 줄여본 뒤 'WSI를 줄이면 어떤지'도 보기 위한 ablation). "
             "기본값(None)과 다르면 wandb/checkpoint에 _EMBDIM{embed_dim} 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--num-transformer-layers", type=int, default=None,
        help="cfg.model.num_transformer_layers(기본 1) 덮어쓰기 — scripts/train_brca_m4.py의 "
             "동명 플래그와 동일 관례(2026-07-19: PAAD에서 2-layer로 시도했다가 표본 대비 "
             "과적합으로 1-layer로 되돌린 전례, config.py 주석 참조). 기본값(None)과 다르면 "
             "wandb/checkpoint에 _VITLAYERS{num_transformer_layers} 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--one-slide-per-case", action="store_true",
        help="케이스당 슬라이드를 대표 1장으로 줄인다(data/dataset.py::_select_representative_slide, "
             "findings_backlog.md 14번 항목). 기본은 미사용(케이스가 가진 슬라이드를 전부 사용하는 "
             "기존 동작) — 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology)는 TCGA는 diagnostic(DX) "
             "WSI 1개/환자, CPTAC는 tumor series 중 최대 용량 1개/case만 쓰는데 우리는 지금까지 "
             "case당 평균 2.5~3.2장을 전부 써왔다 — 그 격차를 좁히는 실험. 켜면 wandb/checkpoint에 "
             "_1SLIDE 접미사가 자동으로 붙는다. (2026-07-21: --external 3시드 검증 결과 M4A/PMA "
             "둘 다 negative result, findings_backlog.md 14번 항목 참조 — --exclude-normal-slides "
             "쪽이 더 유망한 절충안.)",
    )
    parser.add_argument(
        "--tile-augment", action="store_true",
        help="레퍼런스 M4_Train.ipynb::get_train_cached_patch_transform()의 학습 시 타일 "
             "augmentation(RandomHorizontalFlip/VerticalFlip/ColorJitter/GaussianBlur)을 흉내낸다. "
             "--image와 함께 쓰면(권장) 매 epoch 실시간으로 다시 augment+encode하는 진짜 버전 "
             "(train_ds.transform=PATCH_TRANSFORM_AUGMENTED, val/test/external은 항상 증강 없는 "
             "PATCH_TRANSFORM) — scripts/reference_repro_m4.py --tile-augment로 pooled/seed126에서 "
             "검증(0.674->0.711, findings_backlog.md 참조). --image 없이 쓰면(구버전 절충안) "
             "--seed로 고정된 augmentation을 슬라이드당 1벌만 미리 추출해둔 features_aug.pt"
             "(utils/extract_features_augmented.py)를 학습 split에서만 읽는다 — 매 epoch 다른 view가 "
             "아니라 고정된 1벌이라 정규화 효과가 약했던 절충안(null result 확인됨). 켜면 wandb/"
             "checkpoint에 _AUG 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--strong-blur", action="store_true",
        help="2026-08-07: --tile-augment --image(진짜 real-time augmentation)와 함께 사용 — "
             "GaussianBlur만 세게 올린다(kernel_size 3->5, sigma 상한 1.0->2.0, 적용확률 "
             "0.15->0.35). ColorJitter/flip은 그대로 둔다 — 염색강도/색상 정보(H&E stain "
             "intensity)는 건드리지 않고 초점/해상도 계열 증강만 강화하는 게 ColorJitter를 "
             "더 세게 하는 것보다 덜 위험한 레버라는 판단(2026-08-06 논의). --tile-augment "
             "없이 쓰면 무시된다. 켜면 model_prefix에 _BLUR 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--exclude-normal-slides", action="store_true",
        help="확인된 정상 조직 슬라이드만 제외하고 케이스당 나머지는 전부 그대로 둔다"
             "(data/dataset.py::_exclude_normal_slides, findings_backlog.md 14번 항목) — "
             "--one-slide-per-case보다 훨씬 덜 급진적인 절충안(TCGA 평균 슬라이드/case "
             "2.52→2.28, CPTAC 3.22→2.76). 기본은 미사용. 켜면 wandb/checkpoint에 _NONORMAL "
             "접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--dx-only-slides", action="store_true",
        help="2026-08-14: TCGA에서 DX(진단용/영구절편)가 아닌 슬라이드(TS/BS 등 냉동절편)만 "
             "제외하고 케이스당 남은 DX 슬라이드는 전부 그대로 둔다(data/dataset.py::_dx_only_slides). "
             "uni2official 조사에서 발견한 두 confound(DX-only 슬라이드 감소 + 좌표스케일 버그) 중 "
             "좌표스케일 버그 없이(자체 추출 좌표 그대로) DX-only 효과만 분리해서 검증하기 위한 "
             "옵션. CPTAC은 DX/TS 구분 정보가 없어 영향받지 않는다(external은 항상 전체 코호트). "
             "기본은 미사용. 켜면 wandb/checkpoint에 _DXONLY 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--tile-risk-head", action="store_true",
        help="2026-08-14: --PMA 전용. diagnose_pma_wsi_structure.py 실측 — MultiComponentPooling의 "
             "top-k 성분이 attn_weights(entropy~0.999, 사실상 uniform으로 붕괴)로 선정되고 있어 "
             "독립적인 관점이 아니었다. 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology) "
             "MorphologyBurdenPooling을 참고해, top-k 선정을 self.attn과 파라미터를 공유하지 않는 "
             "별도의 단순 TileRiskHead(게이트 없는 얕은 MLP)로 분리한다(models/"
             "multi_component_pooling.py::TileRiskHead). 동시에 레퍼런스의 risk_stats(패치별 risk "
             "점수 분포를 요약하는 10개 스칼라 — mean/std/max/quantile/top05/top10/frac_over_50/"
             "frac_over_70)를 risk_head 입력에 spatial_feat과 나란히 추가한다. 기본은 미사용(기존 "
             "동작 완전히 유지). 켜면 wandb/checkpoint에 _RISKHEAD 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--reference-cohort", action="store_true",
        help="레퍼런스(Leeyoungsup/pancreatic_cancer_pathology)의 케이스 포함 기준(24개월 시점 "
             "생존 여부 확정 + WSI 보유, data/reference_cohort.py::reference_eligible_case_ids "
             "참조)으로 case를 제한한다 — train_light.py --match-reference-cohort와 동일한 "
             "메커니즘을 train.py(WSI 모델)에도 이식. --dataset/--external로 쓰이는 코호트 전부에 "
             "적용되므로(train/val/test/external 공통 ds_kwargs), --external과 함께 쓰면 external "
             "평가 코호트도 함께 줄어든다(레퍼런스가 CPTAC 평가도 같은 205명 풀 안에서 하는 것과 "
             "동일 관례) — 기존 baseline(external 항상 전체 144명)과 직접 비교하려면 이 점을 "
             "감안할 것. 기본은 미사용. 켜면 wandb/checkpoint에 _REFCOHORT 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--stage-stratify", action="store_true",
        help="2026-08-14: train/val/test(및 k-fold) split의 stratification key에 ajcc_stage를 "
             "추가한다(data/dataset.py::WSISurvivalDataset use_stage_stratify, 기본 False로 "
             "기존 동작 유지). fold별 internal log-rank p가 요동친 원인 조사에서, event 비율/"
             "표본 크기는 fold 간 거의 동일한데 stage 구성만 뚜렷이 달랐다(나쁜 fold는 Stage "
             "IIB가 77%까지 쏠림) — 단일 병기에 쏠린 fold는 위험도 스펙트럼이 좁아 log-rank "
             "검정력 자체가 약해진다. 켜면 같은 seed라도 fold 배정 자체가 기존과 달라지므로, "
             "기존 결과와 비교하려면 baseline도 이 옵션으로 다시 돌려야 한다. 켜면 wandb/"
             "checkpoint에 _STGSTRAT 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--leverage-stratify", action="store_true",
        help="2026-08-14: train/val/test(및 k-fold) split의 stratification key에 high_leverage("
             "data/dataset.py::_HIGH_LEVERAGE_CASE_IDS 소속 여부, 기본 False로 기존 동작 유지)를 "
             "추가한다. --stage-stratify로도 fold별 log-rank p 변동이 개별 fold 단위에서는 "
             "깔끔히 설명되지 않아, 다변량 상관 분석에서 가장 강했던 후보(고레버리지 환자 집중도 "
             "rho=0.894)를 직접 통제해보는 탐색적 실험. _HIGH_LEVERAGE_CASE_IDS는 baseline "
             "3seed pooled OOF의 leave-one-out c-index delta로 역산한 20명으로 모델 종속적인 "
             "정의임을 유의(일반적 '어려운 환자' 정답이 아님). 켜면 같은 seed라도 fold 배정 "
             "자체가 기존과 달라지므로, 기존 결과와 비교하려면 baseline도 이 옵션으로 다시 돌려야 "
             "한다. 켜면 wandb/checkpoint에 _LEVSTRAT 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--rna-dim", type=int, default=None,
        help="--PMA 전용: RNA 인코더 출력 차원(기본 None=cfg.model.embed_dim과 동일, 기존 동작). "
             "레퍼런스처럼 RNA를 WSI(embed_dim)보다 넓게 쓰는 조합(예: RNA=128, WSI=64)을 "
             "시도할 때 --clinical-dim과 함께 사용. 기본값과 다르면 wandb/checkpoint에 "
             "_RNADIM{rna-dim} 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--clinical-dim", type=int, default=None,
        help="--PMA 전용: Clinical 인코더 출력 차원(기본 None=cfg.model.embed_dim과 동일). "
             "레퍼런스처럼 Clinical을 좁게 쓰는 조합(예: Clinical=16)을 시도할 때 --rna-dim과 "
             "함께 사용. 기본값과 다르면 wandb/checkpoint에 _CLINDIM{clinical-dim} 접미사가 "
             "자동으로 붙는다.",
    )
    parser.add_argument(
        "--shuffle-patches", action="store_true",
        help="list_patch_paths()가 항상 반환하는 좌표순 고정 순서 대신, 학습 forward마다 패치 "
             "순서를 무작위로 섞는다(coords/features/patch_paths를 같은 permutation으로 함께 "
             "재정렬 — 패치별 좌표 자체는 안 바뀜). NystromAttention의 landmark가 순서대로 연속된 "
             "패치를 그룹핑해 평균내는 방식이라(nystrom_attention 패키지), 고정 순서면 매 epoch "
             "같은 landmark 그룹핑이 반복된다 — 이게 과적합에 기여하는지 검증하는 실험. "
             "--patch-keep-frac<1.0과 별개로 독립 동작(frac=1.0이어도 순서만 섞을 수 있음). "
             "켜면 wandb/checkpoint에 _SHUF 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--rna-gate-only", action="store_true",
        help="--PMA 전용: z_rna를 component_coattn의 query(WSI 4관점 중 고르는 용도)로만 쓰고, "
             "risk_head 직결 concat에서는 뺀다(risk_head 입력이 [z_wsi, z_clinical] 2D로 축소, "
             "findings_backlog.md 최상위 발견 2차) — RNA 정보는 co-attention을 통해 z_wsi에 "
             "여전히 녹아들지만, risk_head가 z_rna로 곧장 우회하는 지름길은 막는다. 켜면 "
             "wandb/checkpoint에 _RNAGATE 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--porpoise-meanpool", action="store_true",
        help="2026-08-31: --PORPOISE 전용 — plain gated-ABMIL(attn_pool)의 patch attention이 "
             "entropy 0.999(거의 완전 uniform)로 나와(scripts/diagnose_porpoise_reliance.py), "
             "PAAD(N≈90)뿐 아니라 BRCA(N≈1058) co-attention heatmap에서도 같은 붕괴가 재현된 "
             "뒤(findings_backlog.md) 나온 ablation — attn_pool을 무파라미터 MeanPooling"
             "(models/vit_porpoise.py::MeanPooling)으로 바꿔, '학습되는 균등 근사'를 진짜 "
             "균등으로 바꿔도 성능이 같은지(=attention 모듈 자체가 불필요한지) 직접 검증한다.",
    )
    parser.add_argument(
        "--porpoise-coattn", action="store_true",
        help="2026-08-31: --PORPOISE 전용, --porpoise-meanpool과 동시 사용 불가 — attn_pool을 "
             "models/vit_m4a.py::CoAttentionPooling(M4A와 동일, RNA가 query인 cross-attention)"
             "으로 바꾼다. '나이스트롬이 patch 간 차이를 뭉개서 RNA co-attention이 구별을 "
             "못 한 것 아니냐'는 가설(--M4A --skip-patch-vit 단독 실험과 별개)을 BilinearFusion "
             "과 결합해서도 확인 — --skip-patch-vit와 같이 쓰면 '나이스트롬 없는 RNA "
             "co-attention + Kronecker fusion' 조합이 된다.",
    )
    parser.add_argument(
        "--porpoise-attn-temperature", type=float, default=1.0,
        help="2026-08-31: --PORPOISE 전용(--porpoise-meanpool/--porpoise-coattn과는 무관, plain "
             "gated-ABMIL에만 적용) — attn_pool의 softmax 이전 score를 이 값으로 나눈다(1보다 "
             "작으면 attention이 더 뾰족해짐, models/vit_m1.py::AttentionPooling 참조). 배경 — "
             "이미 학습된(T=1) 체크포인트에 재학습 없이 낮은 T로 후처리 sharpening을 해보니 "
             "entropy는 낮아졌지만 C-index는 오히려 계속 떨어졌다(diagnose 결과, T=1 전제로 만든"
             "raw score를 재해석하면 신호뿐 아니라 노이즈까지 같이 증폭됨) — 그래서 이번엔 "
             "학습 자체를 낮은 T로 처음부터 하게 만드는 ablation. 1.0(기본)이면 기존과 동일.",
    )
    parser.add_argument(
        "--surv-loss", type=str, default="cox", choices=["cox", "nll_surv", "both"],
        help="2026-09-06: 생존 loss 함수 선택. 기본 'cox'(utils/losses.py::cox_ph_loss, 이 "
             "프로젝트 전체 기본값, risk_head가 스칼라 log-risk 1개를 뱉음)는 동작 변화 없음. "
             "'nll_surv'는 PORPOISE(Chen et al. 2022) 원조 discretized-time NLL(utils/losses.py::"
             "nll_surv_loss, Zadeh&Schmid 2020) — risk_head가 --nll-n-bins개 시간-구간별 raw "
             "hazard logit을 뱉도록 바뀐다(models/vit_porpoise.py::ViT_PORPOISE surv_n_classes). "
             "'PORPOISE 공식 코드를 그대로 재현'(sbatch/run_porpoise_official_paad_*.sh)과는 "
             "별개 실험 — 이쪽은 우리 아키텍처/백본을 그대로 두고 loss 함수만 저쪽 것으로 "
             "바꿔서, loss 함수 차이 자체가 성능에 미치는 영향만 분리해서 본다. "
             "'both'(2026-09-06 추가): 두 loss를 더해서 같이 최적화한다 — cox/nll_surv 매칭 "
             "실험에서 nll_surv가 raw C-index는 근소 우세, cox가 HR/log-rank 유의성·seed 간 "
             "안정성은 확실히 우세했던 관찰(서로 다른 것을 최적화하는 게 원인으로 보임 — Cox "
             "score test는 log-rank와 점근적으로 동일, nll_surv는 이분화 분리력을 직접 밀지 "
             "않음)에서, 한쪽만 고르지 않고 같은 risk_head 출력(hazard logit)에서 둘 다 계산해 "
             "더하면 상호 보완될 수 있다는 아이디어. hazard logit -> utils/losses.py::"
             "hazard_to_risk로 스칼라화한 뒤 그 스칼라에 cox_ph_loss를 추가로 적용(가중치는 "
             "--nll-cox-weight). --PORPOISE/--PMA에서만 쓸 수 있다(다른 모델은 risk_head 출력 "
             "차원 변경 미지원).",
    )
    parser.add_argument(
        "--nll-n-bins", type=int, default=4,
        help="--surv-loss nll_surv/both 전용 — 생존시간을 몇 개 구간으로 이산화할지(PORPOISE "
             "논문 기본값 4). 구간 경계는 매 fold의 train split 내 사망자(OS_event=1) OS_time만"
             "으로 quantile fit한다(utils/losses.py::fit_survival_bins) — PORPOISE 원본은 전체 "
             "코호트로 fit하지만, 이 프로젝트의 RNA 유전자 선정 leakage 전례(findings_backlog.md)"
             "를 피하려고 항상 그 fold의 train split만 쓴다.",
    )
    parser.add_argument(
        "--nll-cox-weight", type=float, default=1.0,
        help="--surv-loss both 전용 — nll_surv_loss + 이 가중치 * cox_ph_loss(같은 hazard "
             "logit에서 유도한 스칼라 risk 기준)로 합산할 때 cox 항의 가중치. 기본 1.0(동등 "
             "가중). 0으로 주면 nll_surv 단독과 사실상 동일(디버그용).",
    )
    parser.add_argument(
        "--no-coattn", action="store_true",
        help="2026-08-31: --PMA 전용 — WSI가 성능에 안 먹히는 원인이 Nystrom self-attention/"
             "ABMIL(MultiComponentPooling attn view)/co-attention 중 무엇인지 분리하는 3종 "
             "ablation의 co-attention 담당(models/vit_pma.py ViT_PMA use_coattn 참조, 나머지 "
             "둘은 기존 --skip-patch-vit/--drop-component attn로 검증). 켜면 component_coattn "
             "자체를 안 만들고, RNA-query 가중합 대신 4개 pooling 관점의 단순 평균을 쓴다 — "
             "RNA가 4관점 중 뭘 볼지 고르는 게 실제로 도움되는지 검증. 켜면 wandb/checkpoint에 "
             "_NOCOATTN 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--cluster-pool", action="store_true",
        help="2026-09-05: --PMA 전용 — Nystrom(oversmoothing 무죄로 확인)과 ABMIL(gradient가 "
             "weight_decay 유무와 무관하게 전혀 안 닿는 dead module로 확인, scripts/"
             "diagnose_abmil_attn_training.py) 둘 다 우회하는 대안. 학습 파라미터 없는 사전계산 "
             "군집 중심(data/cluster_centroids_{backbone}.pt, K=10, raw feature 공간)으로 패치 "
             "N개를 K개의 '슬라이드 내 실존 조직 유형' 대표값으로 미리 요약해, 기존 4-component "
             "(mean/std/attn/top) 자리에 그대로 꽂아 RNA co-attention에 넘긴다(models/vit_pma.py "
             "ViT_PMA.forward cluster_pool 분기). self.vit/self.attn_pool(MultiComponentPooling)은 "
             "생성은 되지만 forward에서 안 쓰인다(파라미터 낭비는 있지만 나머지 코드 경로 호환 "
             "유지). 켜면 wandb/checkpoint에 _CLUSTERPOOL 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--cluster-centroids-path", type=str, default=None,
        help="2026-09-05: --cluster-pool 전용 — 기본(None)이면 data/cluster_centroids_{backbone}.pt. "
             "K/적합 데이터셋을 바꿔가며 비교할 때(예: TCGA-only 재적합 vs 원래 tcga+cptac 적합) "
             "다른 경로를 명시적으로 지정하기 위함.",
    )
    parser.add_argument(
        "--cluster-pool-temperature", type=float, default=None,
        help="2026-09-05: --cluster-pool 전용 — None(기본)이면 hard argmin(패치를 가장 가까운 "
             "군집 1개에 확정 배정). 양수를 주면 -distance/T의 softmax로 모든 K개 군집에 부드럽게 "
             "걸치는 가중평균을 쓴다(fuzzy c-means류, 경계에 걸친 패치의 정보 손실을 줄임). "
             "raw feature 공간 거리 스케일 감(TCGA-only K=11 재적합 기준 RMS 거리 ~14) 참고해 "
             "값을 잡을 것. 켜면 wandb/checkpoint에 _CLUSTERTEMP{T} 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--tumor-content-head-path", type=str, default=None,
        help="2026-09-05: --cluster-pool 전용 — PanNuke(우리 코호트/라벨 완전 미참조)로 학습된 "
             "frozen TumorContentHead(models/tumor_content_head.py) 체크포인트 경로를 주면, "
             "패치별 종양함량 점수(0~1)를 군집 가중치에 곱해 정상/기질 조직이 군집 대표값을 "
             "희석하는 걸 줄인다. 기본(None)이면 미사용(기존 동작 그대로). 예: "
             "data/hdp_pretrain_tumor_content_head.pt. 켜면 wandb/checkpoint에 _TUMORFILTER "
             "접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--cluster-pool-after-vit", action="store_true",
        help="2026-09-05: --cluster-pool과 함께만 의미 있음 — 원본 PMA 구조에서 ABMIL'만' "
             "cluster_pool로 교체(Nystrom은 그대로 살림). raw feature로 군집 배정은 그대로 "
             "정하되, 그 군집별 평균을 raw feature가 아니라 self.vit(Nystrom)를 통과한 "
             "ctx_tokens 위에서 계산한다 — 'Nystrom-클러스터풀-co-attention' 구조 검증용. "
             "켜면 wandb/checkpoint에 _CLUSTERPOOLVIT 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--no-clinical", action="store_true",
        help="2026-07-28: --PMA 전용, M3(WSI+RNA, clinical 제외) ablation용 — clinical_encoder "
             "자체를 안 만들고 risk_head 입력에서 z_clinical을 뺀다(risk_head 입력이 [z_wsi, "
             "z_rna]로 축소, rna_gate_only와도 함께 쓸 수 있음). 켜면 wandb/checkpoint에 "
             "_NOCLINICAL 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--pretrained-wsi-trunk", type=str, default=None,
        help="scripts/pretrain_brca_wsi_trunk.py 산출물 체크포인트 경로. model.cnn(proj)와 "
             "model.vit(Nystromformer)만 이 가중치로 초기화한다 — RNA가 개입하는 부분"
             "(RNAEncoder/rna_aux_head/component_coattn)은 원래대로 새로 초기화돼 이 코호트의 "
             "유전자셋(--rna-genes)으로 학습된다. WSI를 쓰는 모든 --M*(cnn/vit를 가진 모델)에서 "
             "사용 가능. 켜면 wandb/checkpoint에 _PRETRAINED 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--fusion", action="store_true",
        help="LateFusionViT 사용 (ViT+ABMIL + Cluster Histogram). "
             "data/fit_clusters.py 실행으로 cluster_centroids.pt 사전 생성 필요.",
    )
    parser.add_argument(
        "--avgpool", action="store_true",
        help="ViT_M1_AvgPool 사용 — ABMIL(학습되는 gated attention pooling) 대신 학습 파라미터가 "
             "없는 단순 평균 풀링으로 패치→WSI 집계를 대체한다. --M1(기본)에서만 지원, "
             "--M2/--M4/--M4A/--M4B/--PM4/--PMA/--M5/--M6/--M6X/--fusion과 동시 사용 불가.",
    )
    # [Clinical/RNA] --M1/--M2/--M4/--M4A/--M4B/--PM4/--PMA/--M5/--M6/--M6X로 모델 종류 선택 (상호 배타)
    # --M1(기본값): 순수 WSI 모델(ViT_M1, --fusion 지정 시 LateFusionViT)
    # --M2        : ViT_M2 — WSI 임베딩 + Clinical(age/sex) MLP Late Fusion 멀티모달
    # --M4        : ViT_M4 — WSI + Clinical(age/sex) + RNA-seq MLP 3-모달 Late Fusion,
    #               RNA-guided attention pooling(FiLM additive bias, ABMIL 게이트에 적용)
    # --M4A       : ViT_M4A — ViT_M4와 fusion 골격 동일, attn_pool만 genomic-guided
    #               co-attention(MCAT 스타일, z_rna가 query)으로 교체한 ablation
    # --M4B       : ViT_M4B — ViT_M4와 fusion 골격 동일, RNA 개입 지점을 ViT *이전*
    #               (patch token 자체에 FiLM)으로 옮긴 ablation
    # --PM4       : ViT_PM4 — ABMIL 단일 벡터 대신 다성분(mean/std/attn-weighted/top-k) pooling.
    #               RNA는 pooling 이후 post-hoc sigmoid 게이트로 개입(레퍼런스 M4 설계 이식)
    # --PMA       : ViT_PMA — PM4와 동일 다성분 pooling, RNA는 4개 관점에 대한
    #               co-attention(query)으로 개입
    # --M5        : ClinicalOnly — Clinical(age/sex)만 사용, WSI/RNA 없음 (구색용 하한선)
    # --M6        : RNAOnly — RNA-seq만 사용, WSI/Clinical 없음 (구색용 하한선)
    # --M6X       : RNAOnlyExtend — M6와 동일 유전자 입력(339개), 인코더만 레퍼런스 사양
    #               (G -> 256 -> 256, dropout 0.25)으로 확장한 ablation
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--M1", action="store_true",
        help="순수 WSI 모델 사용 (기본값). --fusion과 함께 쓰면 LateFusionViT, "
             "아니면 ViT_M1.",
    )
    model_group.add_argument(
        "--M2", action="store_true",
        help="ViT_M2 사용 (ViT+ABMIL + Clinical(age/sex) MLP Late Fusion 멀티모달). "
             "data/clinical_{tcga,cptac}.csv 필요. --fusion과 동시 사용 불가.",
    )
    model_group.add_argument(
        "--M4", action="store_true",
        help="ViT_M4 사용 (ViT+ABMIL + Clinical(age/sex) MLP + RNA-seq MLP "
             "3-모달 Late Fusion, RNA-guided attention pooling(FiLM)). "
             "data/clinical_{tcga,cptac}.csv, data/rna_{tcga,cptac}.csv "
             "필요. --fusion과 동시 사용 불가.",
    )
    model_group.add_argument(
        "--M4A", action="store_true",
        help="ViT_M4A 사용 (ViT_M4와 동일한 3-모달 Late Fusion 골격에서 attn_pool만 "
             "genomic-guided co-attention(MCAT 스타일, z_rna가 query)으로 교체한 ablation). "
             "data/clinical_{tcga,cptac}.csv, data/rna_{tcga,cptac}.csv 필요. "
             "--fusion과 동시 사용 불가.",
    )
    model_group.add_argument(
        "--M4B", action="store_true",
        help="ViT_M4B 사용 (ViT_M4와 동일한 3-모달 Late Fusion 골격에서, RNA 개입 지점을 "
             "ViT 이전 patch token 자체(FiLM scale+shift)로 옮긴 ablation). "
             "data/clinical_{tcga,cptac}.csv, data/rna_{tcga,cptac}.csv 필요. "
             "--fusion과 동시 사용 불가.",
    )
    model_group.add_argument(
        "--MCAT", action="store_true",
        help="ViT_MCAT 사용(2026-08-31, models/vit_mcat.py) — ViT_M4A(단일 z_rna 벡터가 "
             "query 1개로 co-attention)보다 한 발 더 나간, 진짜 MCAT/SurvPath 스타일 — RNA를 "
             "PDAC 기능별 유전자 8개 카테고리(data/select_rnaseq_genes.py::"
             "PDAC_LITERATURE_GENE_SETS)로 나눠 카테고리마다 학습되는 pathway 토큰을 만들고, "
             "이 8개 토큰이 동시에 patch 토큰 전체에 co-attention한다. attention entropy가 "
             "0.999~1.000으로 붕괴했던 원인(query 1개 vs key 4개짜리 저용량 co-attention, "
             "findings_backlog.md 최상위 발견)을 query 개수 자체를 늘려 정면으로 해소하려는 "
             "시도. data/clinical_{tcga,cptac}.csv, data/rna_{tcga,cptac}.csv 필요. "
             "--fusion과 동시 사용 불가.",
    )
    model_group.add_argument(
        "--PORPOISE", action="store_true",
        help="ViT_PORPOISE 사용(2026-08-31, models/vit_porpoise.py, Phase 3) — MCAT까지 포함해 "
             "attention이 '중요 patch를 찾아내야' 하는 계열(M4/M4A/PMA/MCAT)이 전부 co-attention "
             "entropy 붕괴(0.9998, query 개수·gradient와 무관)로 막힌 뒤의 대안. WSI는 RNA와 "
             "무관한 평범한 gated-ABMIL로 풀링하고, WSI-RNA 상호작용은 풀링 이후 Kronecker/"
             "bilinear product(models/bilinear_fusion.py, Chen et al. 2022 PORPOISE)로 명시적으로 "
             "포착한다 — patch 판별력에 의존하지 않는 구조. combine_mode는 항상 cox_add로 "
             "고정(clinical은 별도 Cox 가산항, --combine-mode 무시). "
             "data/clinical_{tcga,cptac}.csv, data/rna_{tcga,cptac}.csv 필요. "
             "--fusion과 동시 사용 불가.",
    )
    model_group.add_argument(
        "--PM4", action="store_true",
        help="ViT_PM4 사용 (다성분 pooling(mean/std/attn-weighted/top-k) + RNA post-hoc "
             "sigmoid 게이트, 레퍼런스 M3/M4의 Morphology Burden Pooling 이식). "
             "data/clinical_{tcga,cptac}.csv, data/rna_{tcga,cptac}.csv 필요. "
             "--fusion과 동시 사용 불가.",
    )
    model_group.add_argument(
        "--PMA", action="store_true",
        help="ViT_PMA 사용 (PM4와 동일 다성분 pooling, RNA가 4개 관점에 대해 "
             "co-attention query로 개입). data/clinical_{tcga,cptac}.csv, "
             "data/rna_{tcga,cptac}.csv 필요. --fusion과 동시 사용 불가.",
    )
    model_group.add_argument(
        "--M4A_FF", action="store_true",
        help="ViT_M4A_FF 사용 (M4A와 동일, Nystromformer FFN 서브레이어만 제거한 맛보기 "
             "ablation, attention이 만드는 공간 컨텍스트는 유지하고 그 이후 비선형 다듬기만 "
             "없앤다). data/clinical_{tcga,cptac}.csv, data/rna_{tcga,cptac}.csv 필요. "
             "--fusion과 동시 사용 불가.",
    )
    model_group.add_argument(
        "--M2_FF", action="store_true",
        help="ViT_M2_FF 사용 (M2에 RNA를 ViTEncoder FFN 직전 FiLM으로만 개입시키는 맛보기 "
             "ablation, 최종 결합(risk_head 직전 concat)엔 RNA가 직접 노출되지 않고 ABMIL "
             "대신 mean pooling을 쓴다). data/clinical_{tcga,cptac}.csv, "
             "data/rna_{tcga,cptac}.csv 필요. --fusion과 동시 사용 불가.",
    )
    model_group.add_argument(
        "--PMA_FF", action="store_true",
        help="ViT_PMA_FF 사용 (PMA와 동일, Nystromformer FFN 서브레이어만 제거한 맛보기 "
             "ablation - M4A_FF와 같은 논리를 다성분 pooling(PMA) 기준에서 마지막으로 확인). "
             "data/clinical_{tcga,cptac}.csv, data/rna_{tcga,cptac}.csv 필요. "
             "--fusion과 동시 사용 불가.",
    )
    model_group.add_argument(
        "--M5", action="store_true",
        help="ClinicalOnly 사용 (Clinical(age/sex) MLP만, WSI/RNA 없음). "
             "data/clinical_{tcga,cptac}.csv 필요. WSI를 전혀 안 쓰므로 --backbone/--image/"
             "--fusion/--avgpool과 함께 써도 무시된다.",
    )
    model_group.add_argument(
        "--M6", action="store_true",
        help="RNAOnly 사용 (RNA-seq MLP만, WSI/Clinical 없음). "
             "data/rna_{tcga,cptac}.csv 필요. WSI를 전혀 안 쓰므로 --backbone/--image/"
             "--fusion/--avgpool과 함께 써도 무시된다.",
    )
    model_group.add_argument(
        "--M6X", action="store_true",
        help="RNAOnlyExtend 사용 (RNAOnly와 동일 유전자 입력, 인코더 폭만 레퍼런스 사양 "
             "G->256->256, dropout 0.25로 확장). data/rna_{tcga,cptac}.csv 필요. WSI를 전혀 "
             "안 쓰므로 --backbone/--image/--fusion/--avgpool과 함께 써도 무시된다.",
    )
    model_group.add_argument(
        "--M1_POOL", action="store_true",
        help="ViT_M1_Pool 사용 (2026-08-05) — M1(ABMIL 단일 벡터)과 M3/PMA(다성분 pooling+"
             "co-attention) 사이의 pooling 방식 불일치를 통제하기 위한 ablation. ABMIL 대신 "
             "MultiComponentPooling(mean/std/attn-weighted/top-k, PMA와 동일)을 쓰되, RNA/"
             "clinical처럼 query로 쓸 외부 모달리티가 없으므로 학습되는 고정 파라미터([CLS]/"
             "DETR object query류)를 query로 co-attention한다 — \"co-attention 구조 자체\"의 "
             "효과를 \"RNA/clinical이 그걸 guide하는 효과\"와 분리해서 보기 위함. WSI만 사용, "
             "clinical/RNA 없음. --fusion/--avgpool과 동시 사용 불가.",
    )
    model_group.add_argument(
        "--M2_POOL", action="store_true",
        help="ViT_M2_Pool 사용 (2026-08-05) — M1_POOL과 같은 이유로, M2(ABMIL+Late Fusion "
             "concat)와 PMA(다성분 pooling+co-attention) 사이의 pooling 방식 불일치를 통제한다. "
             "M1_POOL의 학습된 고정 query 대신 z_clinical(ClinicalEncoder 출력)을 co-attention "
             "query로 써서 \"clinical이 WSI pooling을 유의미하게 guide하는가\"를 PMA(z_rna가 "
             "query)와 대칭적으로 검증 — margin(--clinical-margin)이 진짜 신호로 확인된 뒤 "
             "처음 시도. --clinical-margin/--no-age-sex 지원(M7/PMA와 동일 관례). "
             "data/clinical_{tcga,cptac}.csv 필요. --fusion/--avgpool과 동시 사용 불가.",
    )
    parser.add_argument(
        "--eval-external-ckpt", type=str, default=None,
        help="2026-08-08: 주어지면 학습을 전혀 하지 않고, 이 경로의 checkpoint를 로드해 --external "
             "코호트 전체에 대해서만 딱 한 번 평가한 뒤 환자별 예측을 .logs/external_preds/에 CSV로 "
             "저장하고 즉시 종료한다(scripts/pool_multiseed_kfold_preds.py의 external 버전 입력용). "
             "다른 모든 인자(--PMA/--backbone/--rna-genes/--fold 등)는 그 checkpoint를 만들 때와 "
             "정확히 똑같이 줘야 한다(모델 구조/차원이 일치해야 state_dict가 로드됨) — 학습 자체를"
             "다시 돌리지 않고 이미 저장된 checkpoint 15개(3seed x 5fold 등)의 external 예측만 "
             "빠르게 재추출할 때 쓴다. --external 없이 쓰면 에러.",
    )
    parser.add_argument(
        "--eval-internal-ckpt", type=str, default=None,
        help="2026-08-30: --eval-external-ckpt의 internal 버전 — 학습을 전혀 하지 않고, 이 경로의 "
             "checkpoint를 로드해 held-out internal test fold에 대해서만 딱 한 번 평가한 뒤 환자별 "
             "예측을 .logs/kfold_preds/에 정상 학습 경로와 완전히 동일한 파일명으로 저장하고 즉시 "
             "종료한다(scripts/pool_multiseed_kfold_preds.py 입력용). internal CSV를 실수로 지웠거나 "
             "재풀링이 필요할 때, 재학습 없이 이미 저장된 checkpoint로부터 복구하는 용도. 다른 모든 "
             "인자는 그 checkpoint를 만들 때와 정확히 똑같이 줘야 한다. --fold 없이 쓰면 에러.",
    )
    parser.add_argument(
        "--eval-soup-ckpts", type=str, default=None,
        help="2026-08-12: [Model soup, Wortsman et al. 2022] 콤마로 구분한 N개 checkpoint 경로를 "
             "받아 state_dict를 파라미터별 단순 평균으로 합친 뒤(재학습 없음) 그 합쳐진 가중치로 "
             "internal(--fold가 주어졌으면 그 fold의 held-out test)과 external(--external이면)을 "
             "각각 딱 한 번 평가하고 예측을 CSV로 저장한다. 보통 같은 fold, 다른 seed로 학습된 "
             "체크포인트들(예: 3seed 같은 fold)을 넣어 '가중치 공간 평균'이 '예측값 평균 앙상블'과 "
             "어떻게 다른지 비교하는 용도. 다른 인자(--PMA/--backbone/--fold 등)는 그 checkpoint를 "
             "만들 때와 동일해야 한다(구조 불일치 시 state_dict 로드 실패).",
    )
    return parser.parse_args()


def _log_line(prefix: str, metrics: dict, td_auc: dict | None = None) -> str:
    """print용 한 줄 로그 문자열 (c_index/HR/log-rank p [+ time-dependent AUC])."""
    line = (
        f"{prefix}_c_index={metrics['c_index']:.4f} | {prefix}_HR={metrics['hr']:.3f} "
        f"[{metrics['hr_ci_lower']:.3f}, {metrics['hr_ci_upper']:.3f}] | "
        f"{prefix}_logrank_p={metrics['log_rank_p']:.4f}"
    )
    if td_auc is not None:
        day_keys = sorted(
            (k for k in td_auc if k.startswith("auc_") and k.endswith("d")),
            key=lambda k: int(k[4:-1]),
        )
        per_day = " ".join(f"{k}={td_auc[k]:.4f}" for k in day_keys)
        line += f" | {prefix}_AUC_mean={td_auc['auc_mean']:.4f}"
        if per_day:
            line += f" ({per_day})"
    return line


def main():
    load_env()
    args   = _parse_args()
    auc_days = tuple(int(x.strip()) for x in args.auc_days.split(",") if x.strip())
    cfg    = Config()
    cfg.data.precomputed = not args.image
    if args.seed is not None:
        cfg.data.seed  = args.seed
        cfg.train.seed = args.seed
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.weight_decay is not None:
        cfg.train.weight_decay = args.weight_decay
    if args.warmup_epochs is not None:
        cfg.train.warmup_epochs = args.warmup_epochs
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.tile_decode_workers is not None:
        cfg.model.tile_decode_workers = args.tile_decode_workers
    if args.patches_root_tcga is not None:
        cfg.data.patches_root_tcga = args.patches_root_tcga
    if args.patches_root_cptac is not None:
        cfg.data.patches_root_cptac = args.patches_root_cptac
    if args.dropout is not None:
        cfg.model.dropout = args.dropout
    if args.embed_dim is not None:
        cfg.model.embed_dim = args.embed_dim
    if args.num_transformer_layers is not None:
        cfg.model.num_transformer_layers = args.num_transformer_layers
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.full_attention:
        cfg.model.use_nystrom = False
    if args.no_spatial_embed:
        cfg.model.use_spatial_embed = False
    if args.rel_bias_attention:
        cfg.model.use_rel_bias_attn = True
        cfg.model.use_nystrom = False
        cfg.model.use_spatial_embed = False
    if args.knn_bias_attention:
        cfg.model.use_knn_bias_attn = True
        cfg.model.knn_attn_k = args.knn_k
        cfg.model.use_nystrom = False
        cfg.model.use_spatial_embed = False
    if args.hybrid_attention:
        cfg.model.use_hybrid_attn = True
        cfg.model.knn_attn_k = args.knn_k
    if args.spatial_autocorr:
        cfg.model.use_spatial_autocorr = True
    if args.attn_dispersion:
        cfg.model.use_attn_dispersion = True
    if args.knn_fixed_bias_attention:
        cfg.model.use_knn_fixed_bias_attn = True
        cfg.model.knn_attn_k = args.knn_k
        cfg.model.knn_bias_tau = args.bias_tau
        cfg.model.knn_bias_learnable_tau = args.learnable_tau
        cfg.model.use_nystrom = False
        cfg.model.use_spatial_embed = False
    if args.fix_nystrom_landmarks:
        cfg.model.fix_nystrom_landmarks = True
    if args.knn_mean_agg:
        cfg.model.use_knn_mean_agg = True
        cfg.model.knn_attn_k = args.knn_k
        cfg.model.use_nystrom = False
        cfg.model.use_spatial_embed = False
    if args.cluster_attn:
        cfg.model.use_cluster_attn = True
        cfg.model.n_clusters = args.n_clusters
        cfg.model.use_nystrom = False
        cfg.model.use_spatial_embed = False

    if args.rna_snn and not args.M6:
        raise ValueError("--rna-snn은 현재 --M6(RNAOnly)에서만 배선돼 있습니다.")
    if args.tumor_type_embed and not args.PMA:
        raise ValueError("--tumor-type-embed는 현재 --PMA에서만 배선돼 있습니다.")

    # [LateFusion] --fusion 플래그 시 cluster_centroids.pt 로드 검증
    if args.fusion and not cfg.data.precomputed:
        raise ValueError("--fusion은 precomputed(features.pt) 모드에서만 지원됩니다. --image와 함께 사용 불가.")
    if args.M2 and args.fusion:
        raise ValueError("--M2(Clinical fusion)와 --fusion(Cluster fusion)은 동시에 지원되지 않습니다.")
    if (args.M4 or args.M4A or args.M4B or args.PM4 or args.PMA or args.M4A_FF or args.M2_FF or args.PMA_FF or args.MCAT or args.PORPOISE) and args.fusion:
        raise ValueError("--M4/--M4A/--M4B/--PM4/--PMA/--MCAT/--PORPOISE(Clinical+RNA fusion)와 --fusion(Cluster fusion)은 동시에 지원되지 않습니다.")
    if (args.M5 or args.M6 or args.M6X) and args.fusion:
        raise ValueError("--M5/--M6/--M6X(WSI-free)와 --fusion(Cluster fusion, WSI 전제)은 동시에 지원되지 않습니다.")
    if args.avgpool and (args.M2 or args.M4A or args.M4B or args.PM4 or args.PMA or args.M4A_FF or args.M2_FF or args.PMA_FF or args.M5 or args.M6 or args.M6X or args.fusion or args.MCAT or args.PORPOISE):
        raise ValueError(
            "--avgpool은 --M1(기본)/--M4에서만 지원됩니다 — "
            "--M2/--M4A/--M4B/--PM4/--PMA/--M5/--M6/--M6X/--fusion/--MCAT/--PORPOISE과 동시 사용 불가."
        )
    if args.clinical_staging and not (
        args.M2 or args.M4 or args.M4A or args.M4B or args.PM4 or args.PMA
        or args.M4A_FF or args.M2_FF or args.PMA_FF or args.M5 or args.M2_POOL or args.MCAT or args.PORPOISE
    ):
        raise ValueError(
            "--clinical-staging은 ClinicalEncoder를 쓰는 모델(--M2/--M4/--M4A/--M4B/--PM4/"
            "--PMA/--M4A_FF/--M2_FF/--M5/--M2_POOL/--MCAT)에서만 사용 가능합니다."
        )
    if args.clinical_mutation and not ((args.M4 and not args.avgpool) or args.PORPOISE or args.PMA):
        # 2026-09-03: models/vit_m4.py::ViT_M4(--M4, --avgpool 미사용)에만 use_mutation/
        # mutation_stats를 이식했다 — 다른 combine_with_clinical_rna 계열(M4A/M4B/PM4)이나
        # ViT_M4_AvgPool은 아직 지원 안 함(_patient_risk의 extra_kwargs["mutation_ord"] 조건부
        # 배선 참조). 2026-09-06: --PORPOISE(ViT_PORPOISE)도 ViT_M4를 상속하고 _clinical_embed를
        # 오버라이드하지 않으므로 동일하게 지원 추가(models/vit_porpoise.py). --PMA도 같은 날
        # 이식(models/vit_pma.py — combine_mode="cox_add"에서만, ViT_PMA 자체 검증 참조).
        raise ValueError("--clinical-mutation은 --M4(--avgpool 미사용)/--PORPOISE/--PMA(combine-mode cox_add)에서만 사용 가능합니다.")
    if (args.rna_dim is not None or args.clinical_dim is not None) and not args.PMA:
        raise ValueError("--rna-dim/--clinical-dim은 --PMA에서만 사용 가능합니다.")
    if args.rna_gate_only and not args.PMA:
        raise ValueError("--rna-gate-only는 --PMA에서만 사용 가능합니다.")
    if args.no_coattn and not args.PMA:
        raise ValueError("--no-coattn은 --PMA에서만 사용 가능합니다.")
    if args.porpoise_meanpool and not args.PORPOISE:
        raise ValueError("--porpoise-meanpool은 --PORPOISE에서만 사용 가능합니다.")
    if args.porpoise_coattn and not args.PORPOISE:
        raise ValueError("--porpoise-coattn은 --PORPOISE에서만 사용 가능합니다.")
    if args.porpoise_meanpool and args.porpoise_coattn:
        raise ValueError("--porpoise-meanpool과 --porpoise-coattn은 동시에 사용할 수 없습니다.")
    if args.porpoise_attn_temperature != 1.0 and not args.PORPOISE:
        raise ValueError("--porpoise-attn-temperature는 --PORPOISE에서만 사용 가능합니다.")
    if args.porpoise_attn_temperature != 1.0 and (args.porpoise_meanpool or args.porpoise_coattn):
        raise ValueError(
            "--porpoise-attn-temperature는 plain gated-ABMIL(--porpoise-meanpool/--porpoise-coattn "
            "둘 다 꺼진 상태)에서만 의미가 있습니다."
        )
    if args.surv_loss in ("nll_surv", "both") and not (args.PORPOISE or args.PMA):
        raise ValueError("--surv-loss nll_surv는 --PORPOISE/--PMA에서만 사용 가능합니다(risk_head "
                          "출력 차원 변경을 이 두 클래스만 지원 — models/vit_porpoise.py, "
                          "models/vit_pma.py).")
    if args.no_clinical and not (args.PMA or args.M4):
        raise ValueError("--no-clinical은 --PMA/--M4에서만 사용 가능합니다.")
    if args.no_clinical and args.M4 and args.combine_mode == "cox_add":
        raise ValueError(
            "--M4 --no-clinical은 --combine-mode cox_add와 함께 쓸 수 없습니다 — "
            "cox_add는 clinical이 있어야 의미가 있는 결합 방식입니다(models/vit_m4.py::ViT_M4 "
            "use_clinical guard 참조). --combine-mode concat(기본)을 쓰세요."
        )
    if args.tile_risk_head and not args.PMA:
        raise ValueError("--tile-risk-head는 --PMA에서만 사용 가능합니다.")
    if args.fusion and args.backbone != "resnet50":
        raise ValueError(
            "--fusion(LateFusionViT)의 cluster_centroids.pt는 ResNet50 raw feature(2048-dim) "
            f"기준으로 사전 계산돼 있어 --backbone {args.backbone}와 호환되지 않습니다. "
            f"{args.backbone}로 --fusion을 쓰려면 data/fit_clusters.py를 해당 backbone의 "
            "feature 파일 기준으로 다시 돌려야 합니다."
        )
    centroids_path = Path(__file__).parent / CENTROIDS_DIR
    if args.fusion and not centroids_path.exists():
        raise FileNotFoundError(
            f"cluster_centroids.pt 없음: {centroids_path}\n"
            "  먼저 실행: python -m data.fit_clusters"
        )
    cluster_centroids = torch.load(centroids_path, map_location="cpu") if args.fusion else None

    # [ExternalTest] --external 플래그 해석: 기본은 미사용(None). 켜져 있으면 --dataset의
    # 반대 코호트를 자동 선택한다(tcga↔cptac). --dataset both는 반대 코호트가 없으므로 에러.
    external_dataset = None
    if args.external:
        if args.dataset == "both":
            raise ValueError(
                "--external은 --dataset both와 함께 쓸 수 없습니다 — "
                "both는 이미 TCGA+CPTAC 전체를 학습에 쓰므로 남는 반대 코호트가 없습니다."
            )
        external_dataset = {"tcga": "cptac", "cptac": "tcga"}[args.dataset]

    if args.rna_genes.endswith("_tcga_only") and not (args.dataset == "tcga" and args.external):
        raise ValueError(
            f"--rna-genes {args.rna_genes}는 --dataset tcga --external(TCGA로 학습 -> "
            "CPTAC 전체를 external test)에서만 의미가 있습니다 — 이 유전자셋은 TCGA train "
            "split만으로 뽑혀 다른 조합(특히 --dataset cptac이나 --dataset both)에서 쓰면 "
            "코호트 불일치로 결과 해석이 잘못됩니다."
        )
    if args.rna_genes.endswith("_cptac_only") and not (args.dataset == "cptac" and args.external):
        # 위 _tcga_only 가드의 반대 방향 버전(2026-08-04, train_light.py와 동일하게 추가).
        raise ValueError(
            f"--rna-genes {args.rna_genes}는 --dataset cptac --external(CPTAC로 학습 -> "
            "TCGA 전체를 external test)에서만 의미가 있습니다 — 이 유전자셋은 CPTAC train "
            "split만으로 뽑혀 다른 조합에서 쓰면 코호트 불일치로 결과 해석이 잘못됩니다."
        )

    # [Clinical] --M2/--M4/--M4A/--M4B/--PM4/--PMA/--M5 시 age z-score 정규화 통계를 학습 코호트
    # (args.dataset)에서 계산해 고정한다(extract_rna_clinical.py의 "데이터셋 내부 z-score
    # 정규화" 관례와 동일). dataset="both"면 두 코호트 clinical.csv를 합쳐 통계를 계산한다.
    if args.M2 or args.M2_POOL or args.M4 or args.M4A or args.M4B or args.PM4 or args.PMA or args.M4A_FF or args.M2_FF or args.PMA_FF or args.M5 or args.MCAT or args.PORPOISE:
        if args.dataset == "both":
            import pandas as pd
            ages = pd.concat([
                pd.read_csv(CLINICAL_PATHS["tcga"])["age_years"],
                pd.read_csv(CLINICAL_PATHS["cptac"])["age_years"],
            ])
            age_mean, age_std = float(ages.mean()), float(ages.std(ddof=0))
        else:
            age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
    else:
        age_mean, age_std = None, None

    # [Staging] --clinical-staging(ClinicalEncoder 입력) 또는 --stage-aux-weight(WSI 보조과제,
    # models/stage_predictor.py::StagePredictionHead) 중 하나라도 켜져 있으면 T/N/M/grade 순서형
    # 정규화 통계가 필요하다 - age_mean/age_std와 동일한 관례로 학습 코호트에서 계산해 고정한다.
    with_staging = args.clinical_staging or args.stage_aux_weight > 0
    if with_staging:
        import pandas as pd
        if args.dataset == "both":
            stage_df = pd.concat([
                pd.read_csv(CLINICAL_PATHS["tcga"]),
                pd.read_csv(CLINICAL_PATHS["cptac"]),
            ])
        else:
            stage_df = pd.read_csv(CLINICAL_PATHS[args.dataset])
        stage_stats = stage_stats_from_df(stage_df)
    else:
        stage_stats = None

    # [Margin] --clinical-margin(ClinicalEncoder 입력) — with_staging과 동일한 관례.
    if args.clinical_margin:
        import pandas as pd
        if args.dataset == "both":
            margin_df = pd.concat([
                pd.read_csv(CLINICAL_PATHS["tcga"])[["residual_disease"]],
                pd.read_csv(CLINICAL_PATHS["cptac"])[["residual_disease"]],
            ])
        else:
            margin_df = pd.read_csv(CLINICAL_PATHS[args.dataset])[["residual_disease"]]
        margin_stats = margin_stats_from_df(margin_df)
    else:
        margin_stats = None

    # [Mutation] --clinical-mutation(ClinicalEncoder 입력) — margin/staging과 동일한 관례
    # (train_light.py --clinical-mutation과 동일).
    if args.clinical_mutation:
        import pandas as pd
        if args.dataset == "both":
            mutation_df = pd.concat([
                pd.read_csv(CLINICAL_PATHS["tcga"]),
                pd.read_csv(CLINICAL_PATHS["cptac"]),
            ])
        else:
            mutation_df = pd.read_csv(CLINICAL_PATHS[args.dataset])
        mutation_stats = mutation_stats_from_df(mutation_df)
    else:
        mutation_stats = None

    # [RNA] --M4/--M4A/--M4B/--PM4/--PMA/--M6/--M6X 시 RNAEncoder 입력 유전자셋을 --rna-genes로
    # 고른다 — 기본(subtype)은 Bailey/Moffitt subtype 분류용 ~340개, literature_{1000,1500,2000}은
    # data/select_rnaseq_genes.py 산출물(생존 예측에 직접 최적화된 유전자셋). WSISurvivalDataset에
    # 그대로 넘겨 실제 로드되는 컬럼과 rna_input_dim이 항상 일치하게 한다.
    uses_rna = args.M4 or args.M4A or args.M4B or args.PM4 or args.PMA or args.M4A_FF or args.M2_FF or args.PMA_FF or args.M6 or args.M6X or args.MCAT or args.PORPOISE
    rna_pathway_categories = None
    mcat_gene_sets = None
    if uses_rna:
        if args.rna_genes == "pathway8":
            rna_pathway_categories = pathway_category_gene_ids()
            rna_gene_ids  = None
            rna_input_dim = len(rna_pathway_categories) + (8 if args.use_cnv else 0)
        elif args.MCAT:
            # [MCAT] GeneGroupEncoder는 rna_gene_ids(개별 유전자 z-score 벡터, 1500개 그대로)와
            # gene_sets(카테고리→유전자ID 매핑)를 둘 다 필요로 한다 — pathway8처럼 미리
            # 카테고리 스칼라로 뭉개면 안 됨(findings_backlog.md "10. Pathway8 집계 — 실패"
            # 참조, models/gene_group_encoder.py 참조). rna_pathway_categories(위 pathway8
            # 분기와 공유하는 변수)에는 절대 대입하지 않는다 — WSISurvivalDataset이
            # rna_pathway_categories is not None이면 개별 유전자 대신 8개 카테고리 합집합
            # (163개)으로 컬럼을 줄여버려(data/dataset.py:789), 모델이 기대하는 1500개
            # gene_ids 인덱스와 실제 rna 텐서 길이가 어긋나 index_select가 범위를 벗어난다
            # (2026-08-31 로컬 스모크테스트에서 CUDA device-side assert로 실제 발견).
            # mcat_gene_sets는 오직 ViT_MCAT(gene_sets=...) 생성자에만 넘긴다.
            rna_gene_ids  = literature_guided_gene_ids_intersection(int(args.rna_genes.split("_")[1]))
            rna_input_dim = len(rna_gene_ids)
            mcat_gene_sets = pathway_category_gene_ids()
        elif args.rna_genes.endswith("_tcga_only") or args.rna_genes.endswith("_cptac_only"):
            # 이 분기를 먼저 안 걸러 아래 일반 분기로 흘려보내면 leaky한 both-결합
            # literature_guided_gene_ids(N)이 조용히 로드된다 — 반드시 여기서
            # single-cohort/FDR 로더로만 보낸다(data/dataset.py::resolve_tcga_only_rna_genes).
            rna_gene_ids  = resolve_tcga_only_rna_genes(args.rna_genes)
            rna_input_dim = len(rna_gene_ids) + (8 if args.use_cnv else 0)
        elif args.rna_genes.endswith("_intersection"):
            rna_gene_ids  = literature_guided_gene_ids_intersection(int(args.rna_genes.split("_")[1]))
            rna_input_dim = len(rna_gene_ids) + (8 if args.use_cnv else 0)
        elif args.rna_genes.startswith("pdac_consistency_"):
            # 2026-09-03: train_light.py --rna-genes pdac_consistency_{500,1000,1500,2000}과
            # 동일 관례 이식 — data/select_rnaseq_genes_pdac_consistency.py 산출물(외부 5개 PDAC
            # 마이크로어레이 데이터셋 교차분석 일관성 |rank| top-N, 우리 코호트/라벨 미참조).
            rna_gene_ids  = pdac_consistency_gene_ids(int(args.rna_genes.rsplit("_", 1)[1]))
            rna_input_dim = len(rna_gene_ids) + (8 if args.use_cnv else 0)
        else:
            rna_gene_ids = (
                pdac_subtype_gene_ids() if args.rna_genes == "subtype"
                else literature_guided_gene_ids(int(args.rna_genes.split("_")[1]))
            )
            rna_input_dim = len(rna_gene_ids) + (8 if args.use_cnv else 0)
    else:
        rna_gene_ids = None
        rna_input_dim = None

    set_seed(cfg.train.seed)
    start_time = datetime.now()
    device = torch.device(cfg.train.device)

    torch.backends.cudnn.benchmark = True

    amp_ctx = _make_amp_ctx()

    if args.M4:
        model_prefix = "M4_AVGPOOL" if args.avgpool else "M4"
    elif args.M4A:
        model_prefix = "M4A"
    elif args.MCAT:
        model_prefix = "MCAT"
    elif args.PORPOISE:
        model_prefix = "PORPOISE"
    elif args.M4B:
        model_prefix = "M4B"
    elif args.PM4:
        model_prefix = "PM4"
    elif args.PMA:
        model_prefix = "PMA"
    elif args.M4A_FF:
        model_prefix = "M4A_FF"
    elif args.M2_FF:
        model_prefix = "M2_FF"
    elif args.PMA_FF:
        model_prefix = "PMA_FF"
    elif args.M5:
        model_prefix = "M5"
    elif args.M6:
        model_prefix = "M6"
    elif args.M6X:
        model_prefix = "M6X"
    elif args.M1_POOL:
        model_prefix = "M1_POOL"
    elif args.M2_POOL:
        model_prefix = "M2_POOL"
    elif args.M2:
        model_prefix = "M2"
    elif args.fusion:
        model_prefix = "M1C"
    elif args.avgpool:
        model_prefix = "M1avg"
    else:
        model_prefix = "M1"
    if args.backbone != "resnet50":
        model_prefix += f"_{args.backbone}"
    if args.rna_genes == "pathway8":
        # _PW8 = 카테고리 평균 pathway 집계(8차원) 사용 표시 — literature_1500(_EX)과는
        # 다른 압축 방식이라 섞이지 않게 별도 접미사를 쓴다.
        model_prefix += "_PW8"
    elif args.rna_genes.endswith("_tcga_only"):
        # _EXT{N} = literature_guided_gene_ids_single_cohort("tcga", N) 사용 표시. 기존 _EX(both
        # 결합, leakage 있음)와 절대 같은 태그를 쓰면 안 된다 — wandb/checkpoint에서 섞이면
        # leaky 버전과 leakage-free 버전 결과를 구분할 수 없게 된다. N까지 태그에 넣어
        # EXT500/EXT1500처럼 서로 다른 크기도 섞이지 않게 한다.
        model_prefix += f"_EXT{args.rna_genes.split('_')[1]}"
    elif args.rna_genes.endswith("_cptac_only"):
        # _tcga_only의 반대 방향 — 같은 spec(예: fdr0.1)이어도 반대 코호트에서 뽑힌 다른
        # 유전자셋이라 CPTAC 접미사로 명시적으로 구분한다(train_light.py와 동일한 관례).
        model_prefix += f"_EXT{args.rna_genes.split('_')[1]}CPTAC"
    elif args.rna_genes.endswith("_intersection"):
        # _INT{n} = TCGA-only/CPTAC-only 순위 교집합(양방향 leakage-free) 사용 표시.
        model_prefix += f"_INT{args.rna_genes.split('_')[1]}"
    elif args.rna_genes.startswith("pdac_consistency_"):
        # _PDACCONS{N} = train_light.py와 동일 관례(JCI Insight 2025 5-데이터셋 교차분석
        # 일관성 순위 top-N, 2026-09-03 이식) — literature_*(_EX) 계열과 절대 안 섞이게 별도 태그.
        model_prefix += f"_PDACCONS{args.rna_genes.rsplit('_', 1)[1]}"
    elif args.rna_genes != "subtype":
        # _EX = literature_guided_gene_ids() 등 확장 유전자셋(레퍼런스 방식) 사용 표시.
        # wandb에서 기본(subtype, ~340개) run과 섞이지 않게 이름/그룹에 항상 붙인다.
        model_prefix += "_EX"
    if args.use_cnv:
        # _CNV = train_light.py와 동일 관례(data/extract_cnv.py 산출물 concat, 2026-09-03 이식).
        model_prefix += "_CNV"
    if args.patch_keep_frac < 1.0:
        # _SS = PatchDropout(패치 서브샘플링) 사용 표시 - 위 _EX와 같은 관례.
        model_prefix += "_SS"
    if args.rna_aux_weight > 0:
        # _AUX = RNA 예측 보조과제(RNAPredictionHead) 사용 표시.
        model_prefix += "_AUX"
    if args.stage_aux_weight > 0:
        # _AUX2 = T-stage/grade 예측 보조과제(StagePredictionHead) 사용 표시 — RNA 보조과제(_AUX)와
        # 구분되는 별도 태그.
        model_prefix += "_AUX2"
    if args.clinical_staging:
        # _STG = ClinicalEncoder 입력에 병기(T/N/M)+grade 추가 사용 표시 — 있음/없음 버전을
        # 둘 다 비교할 수 있게 독립 접미사로 뒀다.
        model_prefix += "_STG"
    if args.clinical_margin:
        # _R = ClinicalEncoder 입력에 절제연 상태(margin) 추가 사용 표시(train_light.py와 동일 관례).
        model_prefix += "_R"
        if args.no_age_sex:
            model_prefix += "_ONLY"
    if args.clinical_mutation:
        # _MUT = train_light.py와 동일 관례(PDAC 4대 driver gene mutation status, 2026-09-03 이식).
        model_prefix += "_MUT"
    if args.exclude_normal_slides:
        # _NONORMAL = 확인된 정상 조직 슬라이드만 제외(케이스당 나머지는 전부 유지) 표시.
        model_prefix += "_NONORMAL"
    if args.dx_only_slides:
        # _DXONLY = TCGA에서 DX(진단용/영구절편)가 아닌 슬라이드만 제외(케이스당 나머지 DX는 전부 유지) 표시.
        model_prefix += "_DXONLY"
    if args.reference_cohort:
        # _REFCOHORT = 레퍼런스의 24개월 시점 생존 확정 + WSI 보유 기준으로 case 제한 표시.
        model_prefix += "_REFCOHORT"
    if args.one_slide_per_case:
        # _1SLIDE = 케이스당 대표 슬라이드 1장만 사용 표시(findings_backlog.md 14번 항목).
        model_prefix += "_1SLIDE"
    if args.tile_augment:
        # _AUG = 학습 시 타일 augmentation(seed 고정, 1회성 features_aug.pt) 사용 표시.
        model_prefix += "_AUG"
        if args.strong_blur:
            model_prefix += "_BLUR"
    if args.dropout is not None and args.dropout != 0.3:
        # _DROP{dropout} = cfg.model.dropout(기본 0.3) 스윕 표시.
        model_prefix += f"_DROP{args.dropout:g}"
    if args.rna_dim is not None:
        model_prefix += f"_RNADIM{args.rna_dim}"
    if args.clinical_dim is not None:
        model_prefix += f"_CLINDIM{args.clinical_dim}"
    if args.embed_dim is not None:
        model_prefix += f"_EMBDIM{args.embed_dim}"
    if args.num_transformer_layers is not None:
        model_prefix += f"_VITLAYERS{args.num_transformer_layers}"
    if args.rna_gate_only:
        model_prefix += "_RNAGATE"
    if args.no_coattn:
        model_prefix += "_NOCOATTN"
    if args.cluster_pool:
        model_prefix += "_CLUSTERPOOL"
    if args.cluster_pool_after_vit:
        model_prefix += "_CLUSTERPOOLVIT"
    if args.cluster_pool_temperature is not None:
        model_prefix += f"_CLUSTERTEMP{args.cluster_pool_temperature:g}"
    if args.tumor_content_head_path is not None:
        model_prefix += "_TUMORFILTER"
    if args.porpoise_meanpool:
        model_prefix += "_MEANPOOL"
    if args.porpoise_coattn:
        model_prefix += "_RNACOATTN"
    if args.porpoise_attn_temperature != 1.0:
        model_prefix += f"_T{args.porpoise_attn_temperature:g}"
    if args.no_clinical:
        model_prefix += "_NOCLINICAL"
    if args.shuffle_patches:
        model_prefix += "_SHUF"
    if args.full_attention:
        model_prefix += "_FULLATTN"
    if args.no_spatial_embed:
        model_prefix += "_NOSPATIAL"
    if args.rel_bias_attention:
        model_prefix += "_RELBIAS"
    if args.knn_bias_attention:
        model_prefix += "_KNNATTN"
    if args.hybrid_attention:
        model_prefix += "_HYBRIDATTN"
    if args.spatial_autocorr:
        model_prefix += "_AUTOCORR"
    if args.attn_dispersion:
        model_prefix += "_DISP"
    if args.knn_fixed_bias_attention:
        model_prefix += "_FIXEDBIAS"
        if args.learnable_tau:
            model_prefix += "_LEARNTAU"
    if args.fix_nystrom_landmarks:
        model_prefix += "_NYSTROMFIX"
    if args.knn_mean_agg:
        model_prefix += "_KNNMEANAGG"
    if args.cluster_attn:
        model_prefix += f"_CLUSTERATTN{args.n_clusters}"
    if args.pretrained_wsi_trunk:
        model_prefix += "_PRETRAINED"
    if args.combine_mode != "concat":
        # _COX_ADD = train_light.py --M7 --combine-mode와 동일 관례.
        model_prefix += f"_{args.combine_mode.upper()}"
    if args.M2_POOL and args.pooling_mode == "selfattn":
        model_prefix += "_SELFATTN"
    if args.drop_component is not None:
        model_prefix += f"_NO{args.drop_component.upper()}"
    if args.top_frac != 0.1:
        model_prefix += f"_TOPFRAC{args.top_frac:g}"
    if args.rna_combine_mode == "cox_add":
        model_prefix += "_RNACOXADD"
    if args.tile_risk_head:
        # _RISKHEAD = top-k 선정을 attn_weights 대신 독립적인 TileRiskHead로 분리 + risk_stats(10개) 추가 표시.
        model_prefix += "_RISKHEAD"
    if args.skip_patch_vit:
        model_prefix += "_NOVIT"
    if args.coord_embed:
        model_prefix += "_COORD"
        if args.coord_embed_concat:
            model_prefix += "_CAT"
        elif args.coord_embed_learnable_scale:
            model_prefix += "_SC"
        if args.coord_embed_shuffle:
            model_prefix += "_SHUF"
    if args.wsi_extra_mlp:
        model_prefix += "_XMLP"
    if args.clinical_lr_mult != 1.0:
        model_prefix += f"_CLR{args.clinical_lr_mult:g}"
    if args.rna_lr_mult != 1.0:
        model_prefix += f"_RLR{args.rna_lr_mult:g}"
    if args.wsi_lr_mult != 1.0:
        model_prefix += f"_WLR{args.wsi_lr_mult:g}"
    if args.lr_mult_warmup_epochs > 0:
        model_prefix += f"_LRMW{args.lr_mult_warmup_epochs}"
    if args.warm_start_clinical:
        model_prefix += "_WSCLIN"
    if args.warm_start_rna:
        model_prefix += "_WSRNA"
    if args.freeze_rna:
        model_prefix += "_FROZRNA"
    if args.auto_branch_balance:
        model_prefix += "_ABB"
    if args.ogm_ge_alpha is not None:
        model_prefix += f"_OGM{args.ogm_ge_alpha:g}"
    if args.modality_dropout_p > 0:
        model_prefix += f"_MODDROP{args.modality_dropout_p:g}"
    if args.entropy_reg_weight > 0:
        model_prefix += f"_ENTREG{args.entropy_reg_weight:g}"
    if args.surv_loss in ("nll_surv", "both"):
        model_prefix += f"_NLLSURV{args.nll_n_bins}"
    if args.surv_loss == "both":
        model_prefix += f"_NLLCOX{args.nll_cox_weight:g}"
    if args.lr is not None:
        model_prefix += f"_LR{args.lr:.0e}"
    if args.weight_decay is not None:
        model_prefix += f"_WD{args.weight_decay:.0e}"
    if args.warmup_epochs is not None:
        model_prefix += f"_WARMUP{args.warmup_epochs}"
    if args.sam:
        # 2026-08-06: 이 태그가 없으면 --sam 유무만 다른 두 실행이 model_prefix/checkpoint/
        # kfold_preds 파일명이 완전히 같아져 서로 덮어쓴다(_AUG/_NOSPATIAL 빠뜨렸을 때와 동일한
        # 사고 클래스, 위 주석 참조) — SAM은 학습 시간이 2배라 특히 재실행 비용이 크다.
        model_prefix += f"_SAM{args.sam_rho:g}"
        if args.sam_wsi_only:
            model_prefix += "_WSIONLY"
    if args.swa:
        model_prefix += "_SWA"
    if args.swad:
        model_prefix += f"_SWAD{args.swad_tolerance:g}"
    if args.early_stop_patience is not None:
        model_prefix += f"_ES{args.early_stop_patience}"
    if args.rna_snn:
        model_prefix += "_RNASNN"
    if args.tumor_type_embed:
        model_prefix += "_TTE"
    if args.stage_stratify:
        model_prefix += "_STGSTRAT"
    if args.leverage_stratify:
        model_prefix += "_LEVSTRAT"
    if args.fold is not None:
        model_prefix += f"_FOLD{args.fold}OF{args.n_folds}"

    # internal(main) run과 external run이 같은 학습 세션임을 알아볼 수 있도록 timestamp를 공유한다.
    run_ts = datetime.now().strftime("%m%d::%H%M")
    # [wandb Group] 모델 종류별로 묶는다 — <모델종류>_<group-ts>. --group-ts를 스윕 스크립트가
    # 넘기면 그 세션의 모든 시드/코호트/internal+external run이 하나의 Group으로 묶이고,
    # 안 넘기면(단발 실행) 이 run 자체의 시작 시각이 group-ts가 돼 그룹 크기가 1이 된다.
    group_ts = args.group_ts or run_ts
    wandb_group = f"{model_prefix}_{group_ts}"
    if WANDB_AVAILABLE:
        run_name = f"{args.dataset.upper()}_{model_prefix}_seed{cfg.train.seed}_{run_ts}"
        wandb.init(
            project="Path-ViT",
            name=run_name,
            group=wandb_group,
            config={
                "epochs":                cfg.train.epochs,
                "lr":                    cfg.train.lr,
                "weight_decay":          cfg.train.weight_decay,
                "seed":                  cfg.train.seed,
                "warmup_epochs":         cfg.train.warmup_epochs,
                "cnn_chunk_size":        cfg.train.cnn_chunk_size,
                "cox_batch_size":        cfg.train.cox_batch_size,
                "embed_dim":             cfg.model.embed_dim,
                "num_heads":             cfg.model.num_heads,
                "num_transformer_layers":cfg.model.num_transformer_layers,
                "dropout":               cfg.model.dropout,
                "num_landmarks":         cfg.model.num_landmarks,
                # [LateFusion/Clinical/RNA] 모델 종류 및 군집 수 기록 — ablation 비교용
                "model":                 ("ViT_M4" if args.M4
                                           else "ViT_M4A" if args.M4A
                                           else "ViT_MCAT" if args.MCAT
                                           else "ViT_PORPOISE" if args.PORPOISE
                                           else "ViT_M4B" if args.M4B
                                           else "ViT_PM4" if args.PM4
                                           else "ViT_PMA" if args.PMA
                                           else "ViT_M4A_FF" if args.M4A_FF
                                           else "ViT_M2_FF" if args.M2_FF
                                           else "ViT_PMA_FF" if args.PMA_FF
                                           else "ClinicalOnly" if args.M5
                                           else "RNAOnly" if args.M6
                                           else "RNAOnlyExtend" if args.M6X
                                           else "ViT_M2" if args.M2
                                           else "LateFusionViT" if args.fusion
                                           else "ViT_M1_AvgPool" if args.avgpool else "ViT_M1"),
                "num_clusters":          int(cluster_centroids.shape[0]) if args.fusion else 0,
                "backbone":              args.backbone,
                "age_mean":              age_mean,
                "age_std":               age_std,
                "rna_input_dim":         rna_input_dim,
                "patch_keep_frac":       args.patch_keep_frac,
                "rna_aux_weight":        args.rna_aux_weight,
                "stage_aux_weight":      args.stage_aux_weight,
                "clinical_staging":      args.clinical_staging,
                "one_slide_per_case":    args.one_slide_per_case,
                "exclude_normal_slides": args.exclude_normal_slides,
                "tile_augment":          args.tile_augment,
                "rna_dim":               args.rna_dim,
                "clinical_dim":          args.clinical_dim,
                "rna_gate_only":         args.rna_gate_only,
                "no_clinical":           args.no_clinical,
                "shuffle_patches":       args.shuffle_patches,
                "full_attention":        args.full_attention,
                "no_spatial_embed":      args.no_spatial_embed,
                "rel_bias_attention":    args.rel_bias_attention,
                "knn_bias_attention":    args.knn_bias_attention,
                "hybrid_attention":      args.hybrid_attention,
                "spatial_autocorr":      args.spatial_autocorr,
                "attn_dispersion":       args.attn_dispersion,
                "knn_fixed_bias_attention": args.knn_fixed_bias_attention,
                "learnable_tau":         args.learnable_tau,
                "fix_nystrom_landmarks": args.fix_nystrom_landmarks,
                "knn_mean_agg":          args.knn_mean_agg,
                "cluster_attn":          args.cluster_attn,
                "n_clusters":            args.n_clusters,
                "dataset":               args.dataset,
                "external_dataset":      external_dataset,
            },
        )

    # 2026-08-14: --stage-aux-weight는 clinical CSV의 staging 필드를 보조과제 타깃으로 읽어야 해서
    # (with_staging=True, 아래 참조), M1처럼 원래 clinical을 아예 join 안 하는 모델에서도 clinical
    # join 자체는 필요하다 — 모델(M1)이 age/sex를 입력으로 쓰진 않지만, 그 join으로 딸려오는
    # staging 컬럼만 aux head가 참조한다(with_staging=True인데 with_clinical=False면
    # data/dataset.py의 검증에서 에러가 났었음).
    with_clinical = (args.M2 or args.M2_POOL or args.M4 or args.M4A or args.M4B or args.PM4 or args.PMA
                      or args.M4A_FF or args.M2_FF or args.PMA_FF or args.M5 or args.stage_aux_weight > 0 or args.MCAT or args.PORPOISE)
    with_rna = args.M4 or args.M4A or args.M4B or args.PM4 or args.PMA or args.M4A_FF or args.M2_FF or args.PMA_FF or args.M6 or args.M6X or args.MCAT or args.PORPOISE

    restrict_case_ids = None
    if args.reference_cohort:
        from data.reference_cohort import reference_eligible_case_ids
        target_datasets = ["tcga", "cptac"] if args.dataset == "both" else [args.dataset]
        if external_dataset:
            target_datasets = list(set(target_datasets) | {external_dataset})
        restrict_case_ids = reference_eligible_case_ids(target_datasets, cfg=cfg.data)
        print(f"--reference-cohort: {len(restrict_case_ids)}개 case로 제한")

    ds_kwargs = dict(
        with_clinical=with_clinical, with_staging=with_staging, with_margin=args.clinical_margin,
        with_mutation=args.clinical_mutation,
        with_rna=with_rna, with_cnv=args.use_cnv, feature_backbone=args.backbone,
        rna_gene_ids=rna_gene_ids, rna_pathway_categories=rna_pathway_categories,
        one_slide_per_case=args.one_slide_per_case,
        exclude_normal_slides=args.exclude_normal_slides,
        dx_only_slides=args.dx_only_slides,
        restrict_case_ids=restrict_case_ids,
        fold=args.fold, n_folds=args.n_folds,
        use_stage_stratify=args.stage_stratify,
        use_leverage_stratify=args.leverage_stratify,
    )
    # --tile-augment는 학습 split에서만 적용한다(val/test/external은 항상 증강 없는 features.pt/
    # PATCH_TRANSFORM). --image와 함께 쓰면 매 epoch 실시간 augmentation(transform 교체),
    # --image 없이 쓰면(precomputed 모드) 기존 1회성 features_aug.pt 폴백.
    train_ds = WSISurvivalDataset(
        cfg.data, dataset=args.dataset, split=("all" if args.full_train else "train"),
        feature_filename_override=(
            FEATURES_AUG_FILENAME if (args.tile_augment and cfg.data.precomputed) else None
        ),
        transform=(
            (PATCH_TRANSFORM_AUGMENTED_CACHED_STRONGBLUR if args.strong_blur else PATCH_TRANSFORM_AUGMENTED_CACHED)
            if (args.tile_augment and not cfg.data.precomputed) else None
        ),
        **ds_kwargs,
    )
    if args.exclude_case_ids:
        exclude_ids = [c.strip() for c in args.exclude_case_ids.split(",") if c.strip()]
        before_n = train_ds.items["case_id"].nunique()
        train_ds.items = train_ds.items[~train_ds.items["case_id"].isin(exclude_ids)].reset_index(drop=True)
        after_n = train_ds.items["case_id"].nunique()
        print(f"--exclude-case-ids: train case {before_n} -> {after_n} ({exclude_ids} 제외)")
    # [진짜 real-time augmentation, 2026-07-22] 디코딩+리사이즈를 학습 시작 시 딱 한 번만 하고
    # RAM에 캐싱 — 레퍼런스(tmp_m1_cell1.py::preload_resized_tile_images)와 동일한 절충
    # (data/patch_utils.py::build_tile_cache 참조). 매 epoch은 이 캐시 위에서 flip/jitter/blur
    # 같은 값싼 augmentation만 새로 적용해 "진짜 매 epoch 다른 augmentation"은 유지한다.
    tile_cache = None
    if args.tile_augment and args.image:
        all_patch_paths = [
            p for i in range(len(train_ds)) for slide in train_ds[i] for p in slide["patch_paths"]
        ]
        print(f"실시간 augmentation용 타일 프리로드: {len(all_patch_paths):,}개 패치")
        tile_cache = build_tile_cache(all_patch_paths, workers=args.tile_decode_workers)
    # [2026-07-23] val/test/external을 features.pt(precomputed)로 강제하던 이전 절충은
    # train(512 리사이즈 raw)과 eval(원본 1024 features.pt) 사이에 유효 배율이 달라지는
    # 버그였다(findings_backlog.md) — eval도 train과 똑같이 512로 리사이즈한 raw 이미지를
    # 그대로 쓴다(PATCH_TRANSFORM_512, 증강 없음). RAM 캐싱은 기본적으로 train만 한다(val/test/
    # external은 매 epoch 전체 재캐싱할 만큼 RAM 여유가 없을 수 있어서) — val은 --cache-val-tiles로
    # 옵트인(아래), test/external은 학습 중 반복 안 되는 1회성 평가라 그대로 디스크 디코딩.
    eval_transform = PATCH_TRANSFORM_512 if (args.tile_augment and args.image) else PATCH_TRANSFORM
    val_ds   = None if args.full_train else WSISurvivalDataset(cfg.data, dataset=args.dataset, split="val",   transform=eval_transform, **ds_kwargs)
    test_ds  = None if args.full_train else WSISurvivalDataset(cfg.data, dataset=args.dataset, split="test",  transform=eval_transform, **ds_kwargs)
    # [2026-08-04] evaluate()가 train_eval/val 둘 다 tile_cache 없이 매 epoch 디스크에서 새로
    # 디코딩하고 있던 게 느린 파일시스템에서 epoch 1도 못 넘기는 병목이었다 — train_eval은 train_ds
    # 와 완전히 같은 데이터라 위에서 만든 tile_cache를 그냥 재사용(추가 메모리 0)하고, val은 별도
    # 캐시가 필요해(--cache-val-tiles) 옵트인으로만 만든다.
    val_tile_cache = None
    if args.tile_augment and args.image and args.cache_val_tiles and val_ds is not None:
        val_patch_paths = [
            p for i in range(len(val_ds)) for slide in val_ds[i] for p in slide["patch_paths"]
        ]
        print(f"--cache-val-tiles: val 타일 프리로드: {len(val_patch_paths):,}개 패치")
        val_tile_cache = build_tile_cache(val_patch_paths, workers=args.tile_decode_workers)
    # train_c_index 리포팅도 항상 증강 없는 eval_transform을 쓴다 — dataset.py의 .transform은
    # __getitem__에서 쓰이지 않고 train.py가 evaluate() 호출 시 명시적으로 넘기는 값이므로
    # (patch_paths/coords 자체는 precomputed 여부만 다르고 train_ds와 동일), 별도 인스턴스 없이
    # train_ds를 그대로 재사용해도 된다.
    train_eval_ds = train_ds
    # [ExternalTest] 학습에 전혀 쓰이지 않은 코호트 전체(split="all") — 없으면 None
    external_ds = (
        WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", transform=eval_transform, **ds_kwargs)
        if external_dataset else None
    )
    # [2026-08-04] val과 동일한 이유(--cache-val-tiles) — external은 보통 run당 1회만 평가되지만
    # (val처럼 매 epoch 반복은 아님), 그 1회가 반대 코호트 전체라 디스크에서 새로 읽으면 여전히
    # 느릴 수 있다. 옵트인(--cache-external-tiles).
    external_tile_cache = None
    if args.tile_augment and args.image and args.cache_external_tiles and external_ds is not None:
        external_patch_paths = [
            p for i in range(len(external_ds)) for slide in external_ds[i] for p in slide["patch_paths"]
        ]
        print(f"--cache-external-tiles: external 타일 프리로드: {len(external_patch_paths):,}개 패치")
        external_tile_cache = build_tile_cache(external_patch_paths, workers=args.tile_decode_workers)

    dl_kwargs = dict(
        batch_size=1,
        collate_fn=_identity_collate,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.data.num_workers > 0),
        prefetch_factor=2 if cfg.data.num_workers > 0 else None,
    )
    train_loader      = DataLoader(train_ds, shuffle=not args.no_patient_shuffle,  **dl_kwargs)
    train_eval_loader = DataLoader(train_eval_ds, shuffle=False, **dl_kwargs)
    val_loader        = DataLoader(val_ds,   shuffle=False, **dl_kwargs) if val_ds  is not None else None
    test_loader       = DataLoader(test_ds,  shuffle=False, **dl_kwargs) if test_ds is not None else None
    external_loader   = DataLoader(external_ds, shuffle=False, **dl_kwargs) if external_ds else None

    # [Clinical/RNA/LateFusion] --M1/--M2/--M4/--M4A/--M4B/--M5/--M6/--fusion에 따라 모델 선택
    # ViT_M1        : 순수 WSI ViT+ABMIL 단일 경로 (--M1, ablation baseline)
    # LateFusionViT : ViT+ABMIL (Path A) + Cluster Histogram (Path B) Late Fusion (--M1 --fusion)
    # ViT_M2        : ViT+ABMIL (WSI) + Clinical age/sex MLP Late Fusion 멀티모달 (--M2)
    # ViT_M4        : ViT+ABMIL (WSI, RNA-guided FiLM) + Clinical age/sex MLP + RNA-seq MLP
    #                 3-모달 Late Fusion (--M4)
    # ViT_M4A       : ViT_M4와 동일 골격, attn_pool만 genomic-guided co-attention(MCAT
    #                 스타일)으로 교체한 ablation (--M4A)
    # ViT_M4B       : ViT_M4와 동일 골격, RNA 개입 지점을 ViT 이전 patch token(FiLM)으로
    #                 옮긴 ablation (--M4B)
    # ViT_PM4       : ABMIL 단일 벡터 대신 다성분 pooling(mean/std/attn-weighted/top-k) +
    #                 RNA post-hoc sigmoid 게이트 (--PM4, 레퍼런스 M3/M4 설계 이식)
    # ViT_PMA       : PM4와 동일 다성분 pooling, RNA가 4개 관점에 co-attention query로 개입 (--PMA)
    # ClinicalOnly  : Clinical(age/sex) MLP만, WSI/RNA 없음 (--M5, 구색용 하한선)
    # RNAOnly       : RNA-seq MLP만, WSI/Clinical 없음 (--M6, 구색용 하한선)
    # RNAOnlyExtend : RNAOnly와 동일 유전자 입력, 인코더 폭만 레퍼런스 사양(G->256->256)으로
    #                 확장 (--M6X)
    stage_kwargs = dict(use_staging=args.clinical_staging, stage_stats=stage_stats)
    # margin_kwargs는 use_margin을 지원하는 ViT_PMA/ClinicalOnly에만 명시적으로 넣는다 — stage_kwargs처럼
    # 전 모델(M4/M4A/M4B/PM4/M4A_FF/M2_FF/PMA_FF)에 **로 뿌리면 use_margin 파라미터가 없는 클래스에서
    # TypeError가 난다.
    margin_kwargs = dict(use_margin=args.clinical_margin, margin_stats=margin_stats,
                          use_age_sex=not args.no_age_sex)
    # mutation_kwargs도 margin_kwargs와 같은 이유로 --M4(avgpool 제외)에만 명시적으로 넣는다
    # (models/vit_m4.py::ViT_M4만 use_mutation/mutation_stats를 받음, 2026-09-03).
    mutation_kwargs = dict(use_mutation=args.clinical_mutation, mutation_stats=mutation_stats)
    if args.init_seed is not None:
        torch.manual_seed(args.init_seed)
    if args.M4 and args.avgpool:
        # 2026-08-14: diagnose_pma_wsi_structure.py 실측 — patch attention entropy~0.999(사실상
        # uniform), attn_pool 자체의 gradient norm이 다른 모듈 대비 100~250배 작음. 이미 사실상
        # 균일하게 동작하는 학습된 attention을, 애초에 학습 파라미터가 없는 평균 풀링으로
        # 교체해 (a) 성능이 정말 동일한지 (b) 무의미한 게이트 파라미터가 유발하는 gradient
        # 노이즈가 없어지면서 seed 간 변동(std)이 줄어드는지 확인한다(models/vit_m4_avgpool.py).
        model = ViT_M4_AvgPool(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                                precomputed=cfg.data.precomputed, backbone=args.backbone,
                                use_attn_dispersion=args.attn_dispersion, combine_mode=args.combine_mode,
                                skip_patch_vit=args.skip_patch_vit,
                                **stage_kwargs, **margin_kwargs).to(device)
    elif args.M4:
        # 2026-08-14: margin(R)/combine_mode(cox_add)/attn_dispersion 이식 — M4A가 2026-08-11에
        # 받은 것과 동일한 업그레이드(models/vit_m4.py 참조). PMA(다성분 pooling+co-attention)와
        # M4(단일 gated-ABMIL, RNA는 attention 게이트에 FiLM으로만 개입)를 같은 레시피로 공정
        # 비교하기 위함 — WSI pooling 구조 하나만 다른 채 나머지 전부(backbone/RNA/clinical/
        # dispersion/cox_add)를 동일하게 맞춘다.
        model = ViT_M4(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                        precomputed=cfg.data.precomputed, backbone=args.backbone,
                        use_attn_dispersion=args.attn_dispersion, combine_mode=args.combine_mode,
                        skip_patch_vit=args.skip_patch_vit, use_clinical=not args.no_clinical,
                        use_coord_embed=args.coord_embed, coord_embed_concat=args.coord_embed_concat,
                        coord_embed_learnable_scale=args.coord_embed_learnable_scale,
                        coord_embed_shuffle=args.coord_embed_shuffle,
                        use_wsi_extra_mlp=args.wsi_extra_mlp,
                        **stage_kwargs, **margin_kwargs, **mutation_kwargs).to(device)
    elif args.M4A:
        # 2026-08-11: margin(R)/combine_mode(cox_add)/attn_dispersion/skip_patch_vit 이식 —
        # patch-level co-attention(MCAT 스타일)을 지금의 최종 레시피와 공정하게 비교하기 위함
        # (models/vit_m4.py 참조. 이전 findings_backlog.md의 M4A 기록은 이 레시피 이전 것들).
        model = ViT_M4A(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                         precomputed=cfg.data.precomputed, backbone=args.backbone,
                         use_attn_dispersion=args.attn_dispersion, combine_mode=args.combine_mode,
                         skip_patch_vit=args.skip_patch_vit,
                         **stage_kwargs, **margin_kwargs).to(device)
    elif args.MCAT:
        # 2026-08-31: M4A(단일 query co-attention)를 진짜 MCAT/SurvPath 스타일 multi-pathway-
        # token co-attention으로 확장 — models/vit_mcat.py 참조.
        model = ViT_MCAT(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                          gene_ids=rna_gene_ids, gene_sets=mcat_gene_sets,
                          precomputed=cfg.data.precomputed, backbone=args.backbone,
                          use_attn_dispersion=args.attn_dispersion, combine_mode=args.combine_mode,
                          skip_patch_vit=args.skip_patch_vit,
                          **stage_kwargs, **margin_kwargs).to(device)
    elif args.PORPOISE:
        # 2026-08-31 Phase 3: attention이 patch를 골라내야 하는 계열(M4/M4A/PMA/MCAT) 전부가
        # co-attention entropy 붕괴로 막힌 뒤의 대안 — models/vit_porpoise.py 참조. combine_mode는
        # 항상 cox_add로 내부 고정이라(clinical은 별도 Cox 가산항) args.combine_mode를 넘기지 않음.
        model = ViT_PORPOISE(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                              precomputed=cfg.data.precomputed, backbone=args.backbone,
                              use_attn_dispersion=args.attn_dispersion,
                              skip_patch_vit=args.skip_patch_vit,
                              use_meanpool=args.porpoise_meanpool, use_coattn=args.porpoise_coattn,
                              attn_temperature=args.porpoise_attn_temperature,
                              surv_n_classes=(args.nll_n_bins if args.surv_loss in ("nll_surv", "both") else 1),
                              **stage_kwargs, **margin_kwargs, **mutation_kwargs).to(device)
    elif args.M4B:
        model = ViT_M4B(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                         precomputed=cfg.data.precomputed, backbone=args.backbone, **stage_kwargs).to(device)
    elif args.PM4:
        model = ViT_PM4(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                         precomputed=cfg.data.precomputed, backbone=args.backbone, **stage_kwargs).to(device)
    elif args.PMA:
        model = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                         precomputed=cfg.data.precomputed, backbone=args.backbone,
                         rna_dim=args.rna_dim, clinical_dim=args.clinical_dim,
                         rna_gate_only=args.rna_gate_only, use_clinical=not args.no_clinical,
                         combine_mode=args.combine_mode, drop_component=args.drop_component,
                         top_frac=args.top_frac, rna_combine_mode=args.rna_combine_mode,
                         skip_patch_vit=args.skip_patch_vit,
                         use_tumor_type_embed=args.tumor_type_embed,
                         use_tile_risk_head=args.tile_risk_head,
                         use_coord_embed=args.coord_embed, coord_embed_concat=args.coord_embed_concat,
                         coord_embed_learnable_scale=args.coord_embed_learnable_scale,
                         coord_embed_shuffle=args.coord_embed_shuffle,
                         use_wsi_extra_mlp=args.wsi_extra_mlp,
                         use_coattn=not args.no_coattn,
                         surv_n_classes=(args.nll_n_bins if args.surv_loss in ("nll_surv", "both") else 1),
                         cluster_pool=args.cluster_pool,
                         cluster_pool_after_vit=args.cluster_pool_after_vit,
                         cluster_pool_temperature=args.cluster_pool_temperature,
                         cluster_centroids_path=args.cluster_centroids_path,
                         tumor_content_head_path=args.tumor_content_head_path,
                         **stage_kwargs, **margin_kwargs, **mutation_kwargs).to(device)
    elif args.M4A_FF:
        model = ViT_M4A_FF(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                            precomputed=cfg.data.precomputed, backbone=args.backbone, **stage_kwargs).to(device)
    elif args.M2_FF:
        model = ViT_M2_FF(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                           precomputed=cfg.data.precomputed, backbone=args.backbone, **stage_kwargs).to(device)
    elif args.PMA_FF:
        model = ViT_PMA_FF(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                            precomputed=cfg.data.precomputed, backbone=args.backbone, **stage_kwargs).to(device)
    elif args.M5:
        model = ClinicalOnly(cfg.model, age_mean=age_mean, age_std=age_std,
                              **stage_kwargs, **margin_kwargs).to(device)
    elif args.M6:
        model = RNAOnly(cfg.model, rna_input_dim=rna_input_dim,
                         rna_encoder_mode="snn" if args.rna_snn else "gelu").to(device)
    elif args.M6X:
        model = RNAOnlyExtend(cfg.model, rna_input_dim=rna_input_dim).to(device)
    elif args.M1_POOL:
        model = ViT_M1_Pool(cfg.model, precomputed=cfg.data.precomputed, backbone=args.backbone,
                             use_attn_dispersion=args.attn_dispersion,
                             use_wsi_extra_mlp=args.wsi_extra_mlp).to(device)
    elif args.M2_POOL:
        model = ViT_M2_Pool(cfg.model, age_mean=age_mean, age_std=age_std,
                             precomputed=cfg.data.precomputed, backbone=args.backbone,
                             use_attn_dispersion=args.attn_dispersion,
                             pooling_mode=args.pooling_mode, combine_mode=args.combine_mode,
                             use_wsi_extra_mlp=args.wsi_extra_mlp,
                             **stage_kwargs, **margin_kwargs).to(device)
    elif args.M2:
        model = ViT_M2(cfg.model, age_mean=age_mean, age_std=age_std,
                        precomputed=cfg.data.precomputed, backbone=args.backbone,
                        use_attn_dispersion=args.attn_dispersion, combine_mode=args.combine_mode,
                        use_margin=args.clinical_margin, margin_stats=margin_stats,
                        skip_patch_vit=args.skip_patch_vit,
                        use_coord_embed=args.coord_embed, coord_embed_concat=args.coord_embed_concat,
                        coord_embed_learnable_scale=args.coord_embed_learnable_scale,
                        coord_embed_shuffle=args.coord_embed_shuffle,
                        use_wsi_extra_mlp=args.wsi_extra_mlp,
                        **stage_kwargs).to(device)
    elif args.fusion:
        model = LateFusionViT(cfg.model, cluster_centroids, precomputed=cfg.data.precomputed).to(device)
    elif args.avgpool:
        model = ViT_M1_AvgPool(cfg.model, precomputed=cfg.data.precomputed, backbone=args.backbone).to(device)
    else:
        # 2026-07-30: use_attn_dispersion — train_multi.py가 M1/M2에도 dispersion을 확장하며
        # ViT_M1/ViT_M2 생성자에 명시적 kwarg로 추가한 것을 여기도 맞춘다. 이전엔 --attn-dispersion이
        # cfg.model.use_attn_dispersion을 True로 세팅해도 M1/M2 생성자가 그 값을 받는 파라미터
        # 자체가 없어 무시됐다(ViT_PMA만 getattr로 읽었음) — M1(기본 모델)에서 처음 실사용 중 발견.
        model = ViT_M1(cfg.model, precomputed=cfg.data.precomputed, backbone=args.backbone,
                        use_attn_dispersion=args.attn_dispersion,
                        skip_patch_vit=args.skip_patch_vit,
                        use_coord_embed=args.coord_embed,
                        coord_embed_concat=args.coord_embed_concat,
                        coord_embed_learnable_scale=args.coord_embed_learnable_scale,
                        coord_embed_shuffle=args.coord_embed_shuffle,
                        use_wsi_extra_mlp=args.wsi_extra_mlp).to(device)
    if args.init_seed is not None:
        torch.manual_seed(cfg.train.seed)
    if hasattr(model, "cnn") and model.cnn.backbone is not None:
        model.cnn.backbone.requires_grad_(False)

    if args.pretrained_wsi_trunk:
        if not (hasattr(model, "cnn") and hasattr(model, "vit")):
            raise ValueError("--pretrained-wsi-trunk는 WSI를 쓰는 모델(cnn/vit 보유)에서만 사용 가능합니다.")
        trunk_ckpt = torch.load(args.pretrained_wsi_trunk, map_location=device, weights_only=False)
        model.cnn.load_state_dict(trunk_ckpt["cnn"])
        model.vit.load_state_dict(trunk_ckpt["vit"])
        print(f"--pretrained-wsi-trunk: {args.pretrained_wsi_trunk}에서 cnn/vit 로드 완료 "
              f"(pretrain epoch={trunk_ckpt.get('epoch')}, val_mse={trunk_ckpt.get('val_mse')})")

    if args.rna_aux_weight > 0:
        if not hasattr(model, "rna_encoder"):
            raise ValueError("--rna-aux-weight는 RNA를 쓰는 모델(--M4/--M4A/--M4B/--PM4/--PMA)에서만 사용 가능합니다.")
        # nn.Module 속성으로 붙이면 PyTorch가 자동으로 서브모듈 등록 -> model.parameters()에
        # 포함됨. optimizer 생성 *이전에* 붙여야 이 헤드의 파라미터도 학습된다.
        model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, rna_input_dim).to(device)

    if args.stage_aux_weight > 0:
        if not hasattr(model, "cnn"):
            raise ValueError("--stage-aux-weight는 WSI를 쓰는 모델에서만 사용 가능합니다 (--M5/--M6/--M6X 불가).")
        # rna_aux_head와 동일한 이유로 optimizer 생성 이전에 붙인다.
        model.stage_aux_head = StagePredictionHead(cfg.model.embed_dim, stage_stats).to(device)

    if args.eval_internal_ckpt:
        # [--eval-internal-ckpt] eval-external-ckpt와 동일 관례, internal held-out fold 버전 —
        # internal CSV를 실수로 지웠을 때 재학습 없이 checkpoint로부터 복구하는 용도(2026-08-30).
        if test_loader is None:
            raise ValueError("--eval-internal-ckpt는 --fold와 함께 써야 합니다(test_loader가 없음).")
        ckpt = torch.load(args.eval_internal_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"--eval-internal-ckpt: {args.eval_internal_ckpt} 로드 완료 "
              f"(epoch={ckpt.get('epoch')}, val_c_index={ckpt.get('val_c_index')})")
        test_metrics = evaluate(model, test_loader, cfg, device, amp_ctx, test_ds.transform,
                                 desc="internal(ckpt-eval)")
        print(f"  internal c_index={test_metrics['c_index']:.4f} | HR={test_metrics['hr']:.3f} "
              f"[{test_metrics['hr_ci_lower']:.3f}, {test_metrics['hr_ci_upper']:.3f}] | "
              f"log_rank_p={test_metrics['log_rank_p']:.4f}")
        import csv
        pred_dir = Path(__file__).parent / ".logs" / "kfold_preds"
        pred_dir.mkdir(parents=True, exist_ok=True)
        # 정상 학습 경로(2963번째 줄 부근)와 완전히 동일한 파일명 — pool_multiseed_kfold_preds.py가
        # 그대로 찾을 수 있어야 하므로 절대 바꾸지 말 것.
        pred_path = pred_dir / f"{args.dataset}_{model_prefix}_seed{cfg.train.seed}_fold{args.fold}of{args.n_folds}.csv"
        with open(pred_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
            for cid, risk, t, e in zip(test_metrics["case_ids"], test_metrics["risks"],
                                        test_metrics["times"], test_metrics["events"]):
                writer.writerow([cid, risk, t, e])
        print(f"  -> internal predictions saved: {pred_path}")
        if hasattr(model, "gene_group_encoder"):
            # [MCAT 진단, 2026-08-31] seed84 fold0 파일럿이 internal C=0.46(<chance)+HR=0.712(<1,
            # 방향 반전)로 나와, 단순히 "구조가 약하다"가 아니라 pathway 토큰/attention이 애초에
            # 붕괴돼 있는 게 아닌지 --eval-internal-ckpt로 복원한 checkpoint에서 직접 확인한다.
            # attention entropy 붕괴는 findings_backlog.md 최상위 발견(4개 view co-attention이
            # 환자 무관하게 0.24~0.27로 수렴)과 동일 패턴이 8개 pathway 토큰에서도 재발했는지 보는 것.
            model.eval()
            z_rna_list, entropy_list = [], []
            with torch.no_grad():
                for patient_slides in test_loader:
                    rna = patient_slides[0]["rna"].to(device, non_blocking=True)
                    z_rna = model.encode_rna(rna)  # (K, D)
                    z_rna_list.append(z_rna.flatten().cpu())
                    slide = patient_slides[0]
                    features = slide.get("features")
                    out = model(
                        slide["coords"].to(device, non_blocking=True),
                        patch_paths=slide.get("patch_paths"),
                        features=features.to(device, non_blocking=True) if features is not None else None,
                        transform=test_ds.transform, rna_context=z_rna,
                    )
                    p = out["attn_weights"].clamp_min(1e-12)
                    entropy_list.append((-(p * p.log()).sum() / torch.log(torch.tensor(float(p.numel())))).item())
            z_stack = torch.stack(z_rna_list)  # (N_patients, K*D)
            z_norm = torch.nn.functional.normalize(z_stack, dim=-1)
            sim = z_norm @ z_norm.T
            n = sim.shape[0]
            off_diag_mean = ((sim.sum() - n) / (n * n - n)).item()
            print("\n=== [MCAT 진단] pathway token 분산 / co-attention entropy ===")
            print(f"  pathway token 환자간 평균 cosine similarity: {off_diag_mean:.4f} "
                  "(1에 가까우면 환자 무관하게 거의 동일한 토큰 = collapse 의심)")
            print(f"  pathway token 환자간 per-dim std 평균: {z_stack.std(dim=0).mean().item():.4f} "
                  "(0에 가까우면 GeneGroupEncoder가 사실상 상수 출력)")
            print(f"  co-attention entropy 평균(0~1 정규화): {sum(entropy_list) / len(entropy_list):.4f} "
                  f"| 범위: [{min(entropy_list):.4f}, {max(entropy_list):.4f}] (1에 가까우면 uniform 붕괴)")
        return

    if args.eval_external_ckpt:
        # [--eval-external-ckpt] 학습을 건너뛰고 이미 저장된 checkpoint의 external 예측만 다시
        # 뽑는다 — model/dataset 생성 코드는 위에서 이미 정상 경로 그대로 실행됐으니(구조 불일치
        # 위험 없음) 여기서 state_dict만 얹고 evaluate()를 한 번 호출한 뒤 즉시 종료한다.
        if external_loader is None:
            raise ValueError("--eval-external-ckpt는 --external과 함께 써야 합니다.")
        ckpt = torch.load(args.eval_external_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"--eval-external-ckpt: {args.eval_external_ckpt} 로드 완료 "
              f"(epoch={ckpt.get('epoch')}, val_c_index={ckpt.get('val_c_index')})")
        external_metrics = evaluate(model, external_loader, cfg, device, amp_ctx, external_ds.transform,
                                     desc="external(ckpt-eval)")
        print(f"  external c_index={external_metrics['c_index']:.4f} | HR={external_metrics['hr']:.3f} "
              f"[{external_metrics['hr_ci_lower']:.3f}, {external_metrics['hr_ci_upper']:.3f}] | "
              f"log_rank_p={external_metrics['log_rank_p']:.4f}")
        import csv
        pred_dir = Path(__file__).parent / ".logs" / "external_preds"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / f"{external_dataset}_{model_prefix}_seed{cfg.train.seed}_fold{args.fold}of{args.n_folds}.csv"
        with open(pred_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
            for cid, risk, t, e in zip(external_metrics["case_ids"], external_metrics["risks"],
                                        external_metrics["times"], external_metrics["events"]):
                writer.writerow([cid, risk, t, e])
        print(f"  -> external predictions saved: {pred_path}")
        return

    if args.eval_soup_ckpts:
        # [Model soup] 서로 다른 seed로 학습된 N개 checkpoint의 state_dict를 파라미터별 단순
        # 평균으로 합쳐(재학습 없음) internal/external을 딱 한 번씩 평가한다.
        ckpt_paths = args.eval_soup_ckpts.split(",")
        state_dicts = [torch.load(p, map_location=device, weights_only=False)["model_state_dict"]
                       for p in ckpt_paths]
        soup_state = {k: sum(sd[k].float() for sd in state_dicts) / len(state_dicts) for k in state_dicts[0]}
        model.load_state_dict(soup_state)
        print(f"--eval-soup-ckpts: {len(ckpt_paths)}개 체크포인트 평균 완료")
        for p in ckpt_paths:
            print(f"    - {p}")

        import csv
        if test_loader is not None:
            soup_test_metrics = evaluate(model, test_loader, cfg, device, amp_ctx, test_ds.transform,
                                          desc="internal(soup-eval)")
            print(f"  internal(soup) c_index={soup_test_metrics['c_index']:.4f}")
            pred_dir = Path(__file__).parent / ".logs" / "kfold_preds"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"{args.dataset}_{model_prefix}_SOUP_fold{args.fold}of{args.n_folds}.csv"
            with open(pred_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
                for cid, risk, t, e in zip(soup_test_metrics["case_ids"], soup_test_metrics["risks"],
                                            soup_test_metrics["times"], soup_test_metrics["events"]):
                    writer.writerow([cid, risk, t, e])
            print(f"  -> internal(soup) predictions saved: {pred_path}")
        if external_loader is not None:
            soup_external_metrics = evaluate(model, external_loader, cfg, device, amp_ctx, external_ds.transform,
                                              desc="external(soup-eval)")
            print(f"  external(soup) c_index={soup_external_metrics['c_index']:.4f}")
            pred_dir = Path(__file__).parent / ".logs" / "external_preds"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"{external_dataset}_{model_prefix}_SOUP_fold{args.fold}of{args.n_folds}.csv"
            with open(pred_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
                for cid, risk, t, e in zip(soup_external_metrics["case_ids"], soup_external_metrics["risks"],
                                            soup_external_metrics["times"], soup_external_metrics["events"]):
                    writer.writerow([cid, risk, t, e])
            print(f"  -> external(soup) predictions saved: {pred_path}")
        return

    # 2026-08-15: clinical_encoder/rna_encoder를 처음부터 공동학습하지 않고 M5(clinical 단독)/
    # M6(RNA 단독)의 이미 수렴한 가중치로 초기화한 뒤 joint fine-tuning을 시작한다 —
    # --clinical-lr-mult/--rna-lr-mult(브랜치가 WSI와 경쟁에서 밀리는 걸 lr로 보정)와 상호보완적인
    # 접근: "0부터 경쟁 시작"이 아니라 "이미 각자 최선인 지점에서 시작"하게 한다.
    if args.warm_start_clinical:
        if not hasattr(model, "clinical_encoder"):
            raise ValueError("--warm-start-clinical인데 이 모델엔 clinical_encoder가 없습니다.")
        ckpt = torch.load(args.warm_start_clinical, map_location=device, weights_only=False)
        prefix = "clinical_encoder."
        sub_state = {k[len(prefix):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith(prefix)}
        model.clinical_encoder.load_state_dict(sub_state)
        print(f"clinical_encoder warm-start: {args.warm_start_clinical}")
    if args.warm_start_rna:
        if not hasattr(model, "rna_encoder"):
            raise ValueError("--warm-start-rna인데 이 모델엔 rna_encoder가 없습니다.")
        ckpt = torch.load(args.warm_start_rna, map_location=device, weights_only=False)
        prefix = "rna_encoder."
        sub_state = {k[len(prefix):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith(prefix)}
        model.rna_encoder.load_state_dict(sub_state)
        print(f"rna_encoder warm-start: {args.warm_start_rna}")
    if args.freeze_rna:
        # 2026-09-03: RNA를 M6 사전학습 가중치로 고정해 "백본처럼" 쓴다(사용자 제안) — 이 파라미터들은
        # requires_grad=False라 옵티마이저 생성(아래 _branch_param_groups 등)에서 자동으로 걸러진다.
        if not args.warm_start_rna:
            raise ValueError("--freeze-rna는 --warm-start-rna와 함께 써야 합니다(무작위 초기화 고정은 무의미).")
        for p in model.rna_encoder.parameters():
            p.requires_grad = False
        print(f"rna_encoder frozen ({sum(p.numel() for p in model.rna_encoder.parameters()):,} params)")

    lr_mult_warmup_targets: list[tuple[int, float]] = []
    if args.sam and args.sam_wsi_only:
        # WSI 브랜치 파라미터만 rho=args.sam_rho, 나머지는 rho=0(=perturbation 없음, 사실상
        # AdamW) — SAM(utils/sam.py)이 param_group마다 다른 rho를 이미 지원해서(first_step의
        # `group["rho"]`) SAM 클래스 자체는 그대로 두고 optimizer 생성 시 param_group만 나눈다.
        _WSI_BRANCH_ATTRS = ("cnn", "vit", "attn_pool", "multi_pool", "component_coattn", "dispersion_scale")
        wsi_params, other_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (wsi_params if name.split(".")[0] in _WSI_BRANCH_ATTRS else other_params).append(p)
        optimizer = SAM(
            [{"params": wsi_params, "rho": args.sam_rho}, {"params": other_params, "rho": 0.0}],
            torch.optim.AdamW, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        )
    elif args.sam:
        optimizer = SAM(
            filter(lambda p: p.requires_grad, model.parameters()),
            torch.optim.AdamW, rho=args.sam_rho,
            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        )
    elif args.clinical_lr_mult != 1.0 or args.rna_lr_mult != 1.0 or args.wsi_lr_mult != 1.0:
        # 2026-08-15: scripts/diagnose_m2_branch_swap.py(fold1, 정상 학습된 체크포인트) 실측 —
        # M2 공동학습 후 clinical_encoder를 M5(clinical 단독)의 risk_head로 채점하면 M5 네이티브
        # 대비 internal -0.075/external -0.018로 떨어지는 반면, WSI 브랜치는 M1 네이티브 대비
        # 거의 안 상한다(+0.021/-0.005). clinical_encoder/clinical_linear(파라미터 수가 WSI
        # 브랜치보다 훨씬 적음)가 같은 lr로 WSI와 경쟁하면 밀려난다는 가설 — --clinical-lr-mult
        # 20배로 fold1 internal 0.4433->0.5355 확인. rna_encoder/rna_linear에도 같은 논리가
        # 적용될 수 있어(M3/M4) --rna-lr-mult로 동일하게 지원 — 여러 브랜치를 동시에 다른
        # 배율로 올릴 수 있게 일반화(--sam-wsi-only와 동일 관례로 param_group 분리).
        branch_groups = _branch_param_groups(model)
        # 2026-09-03: 심각한 기존 버그 발견 — mult==1.0인 브랜치는 아래 for 루프에서 그냥
        # continue돼 어떤 param_group에도 안 들어갔다. clinical/rna/wsi가 전부 "other"와
        # 분리된 별도 키라(_branch_param_groups), 예를 들어 --clinical-lr-mult만 켜고
        # --rna-lr-mult/--wsi-lr-mult는 기본값(1.0)으로 두면 rna_encoder/wsi 파라미터가
        # 옵티마이저에 아예 등록되지 않아 그 브랜치가 학습 내내 무작위 초기화 상태로 방치됐다
        # (오늘 --clinical-lr-mult 5/10/20이 배율과 무관하게 전부 거의 동일하게 나쁜 external로
        # 수렴한 게 이 버그 때문이었음 — rna_encoder가 셋 다 안 학습됐던 것). mult==1.0인
        # 브랜치는 전부 base(그룹 lr) 쪽에 합쳐 넣어 "배율 1.0 = 평소처럼 학습"이 되게 고쳤다.
        base_params = list(branch_groups["other"])
        for key, mult in (
            ("clinical", args.clinical_lr_mult), ("rna", args.rna_lr_mult), ("wsi", args.wsi_lr_mult),
        ):
            if mult == 1.0:
                base_params += branch_groups[key]
        param_groups = [{"params": base_params, "lr": cfg.train.lr}]
        # 2026-08-15: --lr-mult-warmup-epochs용 — 어느 param_group index가 어떤 배율을 목표로
        # 하는지 기억해 둔다(매 epoch 시작 시 실제 배율을 1.0->목표까지 선형으로 올리는 데 사용).
        for key, mult in (
            ("clinical", args.clinical_lr_mult), ("rna", args.rna_lr_mult), ("wsi", args.wsi_lr_mult),
        ):
            if mult == 1.0:
                continue
            if not branch_groups[key]:
                raise ValueError(f"--{key}-lr-mult != 1.0인데 {_BRANCH_ATTRS[key]} 파라미터가 없는 모델입니다.")
            param_groups.append({"params": branch_groups[key], "lr": cfg.train.lr * mult})
            lr_mult_warmup_targets.append((len(param_groups) - 1, mult))
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        )
    scheduler = _build_scheduler(optimizer, cfg)
    auto_balance_groups = (
        _branch_param_groups(model) if (args.auto_branch_balance or args.ogm_ge_alpha is not None) else None
    )

    mode = "precomputed features" if cfg.data.precomputed else "raw image (--image)"
    print(f"Mode: {mode}")
    # [Clinical/RNA/LateFusion] 모델 종류 출력
    if args.M4 and args.avgpool:
        print(f"Model: ViT_M4_AvgPool (ViT+무학습 평균 풀링(attention 제거) + Clinical age/sex MLP + RNA-seq MLP, "
              f"age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.M4:
        print(f"Model: ViT_M4 (ViT+ABMIL(RNA-guided FiLM) + Clinical age/sex MLP + RNA-seq MLP, "
              f"age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.M4A:
        print(f"Model: ViT_M4A (ViT+CoAttentionPooling(RNA query) + Clinical age/sex MLP + RNA-seq MLP, "
              f"age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.MCAT:
        print(f"Model: ViT_MCAT (ViT+MultiQueryCoAttentionPooling(8개 pathway token query) + "
              f"Clinical age/sex MLP + GeneGroupEncoder, "
              f"age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.PORPOISE:
        # 2026-08-31: --porpoise-meanpool/--porpoise-coattn과 무관하게 항상 "gated-ABMIL"로
        # 고정 출력되던 버그 수정 — 실제 모델 구성(models/vit_porpoise.py)은 세 옵션 다 맞게
        # 갈라졌지만, 로그의 자기 설명이 실제 attn_pool 종류와 달라 혼란을 줄 수 있었다.
        pooling_desc = (
            "MeanPooling(무파라미터)" if args.porpoise_meanpool
            else "CoAttentionPooling(RNA query, M4A와 동일 클래스)" if args.porpoise_coattn
            else "gated-ABMIL(RNA 무관)"
        )
        print(f"Model: ViT_PORPOISE (ViT+{pooling_desc} + BilinearFusion(Kronecker product, "
              f"WSI×RNA) + Clinical cox_add, "
              f"age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.M4B:
        print(f"Model: ViT_M4B (ViT+pre-ViT FiLM(RNA) token conditioning + Clinical age/sex MLP + "
              f"RNA-seq MLP, age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.PM4:
        print(f"Model: ViT_PM4 (ViT+다성분 pooling(mean/std/attn/top-k) + RNA post-hoc gate + "
              f"Clinical age/sex MLP, age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.PMA:
        pooling_combine_desc = ("CoAttention(RNA query, 4개 관점)" if not args.no_coattn
                                 else "4개 관점 단순 평균(co-attention 없음)")
        print(f"Model: ViT_PMA (ViT+다성분 pooling + {pooling_combine_desc} + "
              f"Clinical age/sex MLP, age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.M4A_FF:
        print(f"Model: ViT_M4A_FF (M4A에서 Nystromformer FFN 서브레이어 제거, CoAttentionPooling(RNA query) + "
              f"Clinical age/sex MLP, age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.M2_FF:
        print(f"Model: ViT_M2_FF (M2 + RNA를 ViTEncoder FFN 직전 FiLM으로만 개입, mean pooling, "
              f"최종 결합엔 RNA 미노출, age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.PMA_FF:
        print(f"Model: ViT_PMA_FF (PMA에서 Nystromformer FFN 서브레이어 제거, 다성분 pooling + "
              f"CoAttention(RNA query, 4개 관점) + Clinical age/sex MLP, age_mean={age_mean:.1f}, "
              f"age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.M5:
        print(f"Model: ClinicalOnly (Clinical age/sex MLP만, WSI/RNA 없음, "
              f"age_mean={age_mean:.1f}, age_std={age_std:.1f})")
    elif args.M6:
        print(f"Model: RNAOnly (RNA-seq MLP만, WSI/Clinical 없음, rna_input_dim={rna_input_dim})")
    elif args.M6X:
        print(f"Model: RNAOnlyExtend (RNA-seq MLP(G->256->256, dropout 0.25)만, WSI/Clinical 없음, "
              f"rna_input_dim={rna_input_dim})")
    elif args.M1_POOL:
        print(f"Model: ViT_M1_Pool (ViT+다성분 pooling + CoAttention(학습된 고정 query), WSI 단독)")
    elif args.M2_POOL:
        pooling_desc = ("CoAttention(Clinical query)" if args.pooling_mode == "coattn"
                         else "SelfAttention(clinical 미개입)")
        combine_desc = ("Clinical MLP concat" if args.combine_mode == "concat"
                         else "Clinical raw feature cox_add")
        print(f"Model: ViT_M2_Pool (ViT+다성분 pooling + {pooling_desc} + {combine_desc}, "
              f"age_mean={age_mean:.1f}, age_std={age_std:.1f})")
    elif args.M2:
        print(f"Model: ViT_M2 (ViT+ABMIL + Clinical age/sex MLP, "
              f"age_mean={age_mean:.1f}, age_std={age_std:.1f})")
    elif args.fusion:
        K = int(cluster_centroids.shape[0])
        print(f"Model: LateFusionViT (ViT+ABMIL + ClusterHistogram, K={K})")
    elif args.avgpool:
        print(f"Model: ViT_M1_AvgPool (ViT + 무학습 평균 풀링, ABMIL 제거)")
    else:
        print(f"Model: ViT_M1 (ViT+ABMIL baseline)")
    if args.full_train:
        print(f"Dataset: {args.dataset}  (--full-train: 6:2:2 split 없이 코호트 전체를 train으로 사용)  "
              f"Train: {len(train_ds)} patients (val/internal test 없음)")
    else:
        print(f"Dataset: {args.dataset}  (6:2:2 stratified split)  "
              f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test(internal): {len(test_ds)} patients")
    if external_ds is not None:
        print(f"External test dataset: {external_dataset}  (전체 코호트, 학습에 미사용)  "
              f"n={len(external_ds)} patients")
    else:
        print("External test: 사용 안 함 (켜려면 --external 지정)")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"AMP=bfloat16 | batch={cfg.train.cox_batch_size} patients (Cox risk set 단위) "
        f"| cnn_chunk={cfg.train.cnn_chunk_size} | workers={cfg.data.num_workers} "
        f"| tile_decode_workers={cfg.model.tile_decode_workers}"
    )
    ckpt_dir  = Path(__file__).parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # [Clinical/RNA/LateFusion] 모델 종류·backbone별로 별도 checkpoint 저장 — ablation 결과 보존.
    # backbone을 태그에 안 넣으면 --backbone uni/resnet50을 오가며 돌릴 때 같은 파일을 덮어써서
    # 서로 다른 feature 차원의 checkpoint가 섞여버린다.
    tag = args.dataset if args.backbone == "resnet50" else f"{args.dataset}_{args.backbone}"
    tag += f"_seed{cfg.train.seed}"
    if args.rna_genes == "pathway8":
        tag += "_PW8"
    elif args.rna_genes.endswith("_tcga_only"):
        # model_prefix와 동일한 이유로 _EX(leaky, both-결합)와 절대 섞이면 안 됨.
        tag += f"_EXT{args.rna_genes.split('_')[1]}"
    elif args.rna_genes.endswith("_cptac_only"):
        # model_prefix와 동일한 이유로 tcga_only 버전과도 섞이면 안 됨.
        tag += f"_EXT{args.rna_genes.split('_')[1]}CPTAC"
    elif args.rna_genes.endswith("_intersection"):
        tag += f"_INT{args.rna_genes.split('_')[1]}"
    elif args.rna_genes.startswith("pdac_consistency_"):
        tag += f"_PDACCONS{args.rna_genes.rsplit('_', 1)[1]}"
    elif args.rna_genes != "subtype":
        # gene set이 다르면 같은 모델 종류라도 입력 차원이 달라 checkpoint가 호환되지 않는다 —
        # backbone 태그와 같은 이유로 파일명에 반드시 구분자를 남긴다.
        tag += "_EX"
    if args.use_cnv:
        tag += "_CNV"
    if args.patch_keep_frac < 1.0:
        tag += "_SS"
    if args.rna_aux_weight > 0:
        tag += "_AUX"
    if args.stage_aux_weight > 0:
        tag += "_AUX2"
    if args.clinical_staging:
        tag += "_STG"
    if args.clinical_margin:
        tag += "_R"
        if args.no_age_sex:
            tag += "_ONLY"
    if args.clinical_mutation:
        tag += "_MUT"
    if args.pretrained_wsi_trunk:
        # 이게 없으면 같은 레시피를 --pretrained-wsi-trunk 유무만 다르게 동시에 돌릴 때
        # 두 프로세스가 같은 checkpoint 파일을 공유해(경합 상태) best checkpoint를 서로
        # 덮어써버린다 — 2026-07-22 실제로 이 버그로 baseline/pretrained 두 run의 최종
        # test/external 지표가 완전히 동일하게 나온 것을 발견해 추가.
        tag += "_PRETRAINED"
    # [체크포인트 충돌 재발 방지, 2026-07-23] 위 수동 목록이 --tile-augment(_AUG)/
    # --no-spatial-embed(_NOSPATIAL) 등을 빠뜨려, 오늘 밤 M7을 처음 넘긴 real-time augment
    # 체크포인트가 이후 --no-spatial-embed 단독 실행에 그대로 덮어써진 사고가 있었다
    # (findings_backlog.md). model_prefix(위, wandb run 이름용)는 모든 플래그를 빠짐없이
    # 반영해온 만큼, 수동 목록을 유지보수하는 대신 아예 model_prefix를 tag에 그대로 붙여
    # 이 버그 클래스 자체를 봉쇄한다(다소 중복되지만 파일명이 길어지는 것뿐, 안전이 우선).
    tag += f"_{model_prefix}"
    if args.M4:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_clinical_rna.pt"
    elif args.M4A:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_clinical_rna_coattn.pt"
    elif args.MCAT:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_mcat.pt"
    elif args.PORPOISE:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_porpoise.pt"
    elif args.M4B:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_clinical_rna_film.pt"
    elif args.PM4:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_pm4.pt"
    elif args.PMA:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_pma.pt"
    elif args.M4A_FF:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_m4a_ff.pt"
    elif args.M2_FF:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_m2_ff.pt"
    elif args.PMA_FF:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_pma_ff.pt"
    elif args.M5:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_clinical_only.pt"
    elif args.M6:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_rna_only.pt"
    elif args.M6X:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_rna_only_extend.pt"
    elif args.M1_POOL:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_m1_pool.pt"
    elif args.M2_POOL:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_m2_pool.pt"
    elif args.M2:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_clinical.pt"
    elif args.fusion:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_fusion.pt"
    elif args.avgpool:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_avgpool.pt"
    else:
        ckpt_path = ckpt_dir / f"survival_{tag}_best.pt"

    patch_subsample_generator = None
    if args.patch_subsample_seed is not None:
        patch_subsample_generator = torch.Generator(device="cpu")
        patch_subsample_generator.manual_seed(args.patch_subsample_seed)

    # [SWA] 후반부(--swa-start-frac 이후) 매 epoch 가중치를 평균 내는 별도 모델을 유지한다.
    # AveragedModel은 forward만 위임하고 encode_rna/combine_with_clinical_rna 등 커스텀 메서드는
    # 안 갖고 있으므로, 평가 시에는 swa_model이 아니라 swa_model.module(평균된 원본 클래스 인스턴스)을
    # 써야 한다. LayerNorm만 쓰는 아키텍처라 BatchNorm running-stats 재계산(update_bn)은 불필요.
    swa_model = torch.optim.swa_utils.AveragedModel(model) if args.swa else None
    swa_start_epoch = round(cfg.train.epochs * args.swa_start_frac) if args.swa else None

    # [SWAD] epoch마다 (val c_index, CPU 가중치 스냅샷)을 모아뒀다가, 학습이 끝난 뒤 val 성능이
    # best epoch 근방 --swad-tolerance 이내로 유지되는 연속 구간만 골라 평균한다(아래 학습 루프
    # 종료 직후 블록 참조). 매 epoch state_dict를 CPU에 복사해두므로 --swa의 AveragedModel과
    # 메모리 사용량이 비슷한 자릿수(모델 1개분 x epoch 수)다.
    swad_epoch_val_c = []
    swad_epoch_states = []

    # --surv-loss nll_surv(PORPOISE 원조 discretized-hazard NLL): 시간-구간 경계를 이 fold의
    # train split(train_ds.items, 케이스당 1행으로 중복 제거)의 OS_time/OS_event만으로 딱 한 번
    # fit한다 — PORPOISE 원본은 전체 코호트(train+val 합쳐서 fold 나누기 전)로 fit하지만,
    # RNA 유전자 선정에서 겪은 것과 같은 종류의 leakage(findings_backlog.md)를 피하려고 이
    # 프로젝트에서는 항상 그 fold의 train split만 쓴다(utils/losses.py::fit_survival_bins).
    nll_bin_edges = None
    if args.surv_loss in ("nll_surv", "both"):
        _train_labels = train_ds.items.drop_duplicates("case_id")
        nll_bin_edges = fit_survival_bins(
            _train_labels["OS_time"].to_numpy(), _train_labels["OS_event"].to_numpy(),
            n_bins=args.nll_n_bins,
        )
        print(f"[nll_surv] train split {len(_train_labels)}명 기준 시간-구간 경계({args.nll_n_bins}bins): "
              f"{nll_bin_edges}")

    best_score   = -1.0
    best_metrics = {}
    epochs_since_improve = 0
    for epoch in range(cfg.train.epochs):
        lr_now        = optimizer.param_groups[0]["lr"]
        if args.lr_mult_warmup_epochs > 0 and lr_mult_warmup_targets:
            # 매 epoch 시작 시, 이미 scheduler.step()이 반영된 "other" 그룹의 현재 lr(lr_now =
            # base_lr * warmup/cosine factor)을 기준으로, 배율 자체를 1.0->목표까지 이 epoch
            # 진행도만큼 선형 보간해 boosted param_group의 lr을 다시 계산한다 — 기존 warmup/cosine
            # 스케줄과 배율-warmup을 곱셈으로 합성.
            progress = min((epoch + 1) / args.lr_mult_warmup_epochs, 1.0)
            for group_idx, target_mult in lr_mult_warmup_targets:
                effective_mult = 1.0 + (target_mult - 1.0) * progress
                optimizer.param_groups[group_idx]["lr"] = lr_now * effective_mult
        loss          = train_one_epoch(model, train_loader, optimizer, cfg, device, amp_ctx, train_ds.transform,
                                         patch_keep_frac=args.patch_keep_frac, rna_aux_weight=args.rna_aux_weight,
                                         stage_aux_weight=args.stage_aux_weight,
                                         shuffle_patches=args.shuffle_patches, tile_cache=tile_cache,
                                         patch_subsample_generator=patch_subsample_generator,
                                         modality_dropout_p=args.modality_dropout_p,
                                         branch_groups=auto_balance_groups,
                                         auto_balance_enabled=args.auto_branch_balance,
                                         ogm_ge_alpha=args.ogm_ge_alpha,
                                         ogm_ge_epoch_progress=epoch / max(cfg.train.epochs - 1, 1),
                                         entropy_reg_weight=args.entropy_reg_weight,
                                         surv_loss=args.surv_loss, nll_bin_edges=nll_bin_edges,
                                         nll_cox_weight=args.nll_cox_weight,
                                         desc=f"epoch {epoch+1} train")
        # train_c_index는 진단용 리포팅일 뿐 학습 신호가 아니라, val/test/external과 동일하게
        # 항상 증강 없는 eval_transform으로 평가한다 — train_ds.transform을 쓰면 --tile-augment
        # --image일 때 매 epoch 학습 91명을 두 번(학습+리포팅) 실시간 augment하게 돼 시간이
        # 배로 든다(2026-07-22 발견, 실측 epoch당 소요가 예상의 2배 가까이 나온 원인).
        train_metrics = evaluate(model, train_eval_loader, cfg, device, amp_ctx, eval_transform, tile_cache=tile_cache,
                                  desc=f"epoch {epoch+1} train_eval")
        scheduler.step()
        if swa_model is not None and (epoch + 1) >= swa_start_epoch:
            swa_model.update_parameters(model)

        if val_ds is None:
            # --full-train: val 자체가 없음(코호트 전체를 train으로 씀) — train 지표만 기록한다.
            print(f"Epoch {epoch+1:3d} | lr={lr_now:.2e} | loss={loss:.4f} | "
                  f"train_c_index={train_metrics['c_index']:.4f}")
            if WANDB_AVAILABLE:
                wandb.log({
                    "train/loss":       loss,
                    "train/lr":         lr_now,
                    "train/c_index":    train_metrics["c_index"],
                    "train/hr":         train_metrics["hr"],
                    "train/log_rank_p": train_metrics["log_rank_p"],
                }, step=epoch + 1)
            continue

        metrics       = evaluate(model, val_loader, cfg, device, amp_ctx, val_ds.transform, tile_cache=val_tile_cache,
                                  desc=f"epoch {epoch+1} val")
        val_td_auc    = compute_time_dependent_auc(
            train_metrics["times"], train_metrics["events"],
            metrics["times"], metrics["events"], metrics["risks"],
            eval_days=auc_days,
        )

        c_index = metrics.get("c_index", float("nan"))
        score   = c_index if not math.isnan(c_index) else -1.0
        if args.swad and not math.isnan(c_index):
            swad_epoch_val_c.append(c_index)
            swad_epoch_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(
            f"Epoch {epoch+1:3d} | lr={lr_now:.2e} | loss={loss:.4f} | "
            f"train_c_index={train_metrics['c_index']:.4f} | " + _log_line("val", metrics, val_td_auc)
        )

        if WANDB_AVAILABLE:
            log_dict = {
                "train/loss":              loss,
                "train/lr":                lr_now,
                "train/c_index":           train_metrics["c_index"],
                "train/hr":                train_metrics["hr"],
                "train/log_rank_p":        train_metrics["log_rank_p"],
                "val_performance/c_index":       metrics["c_index"],
                "val_performance/hr":            metrics["hr"],
                "val_performance/hr_ci_lower":   metrics["hr_ci_lower"],
                "val_performance/hr_ci_upper":   metrics["hr_ci_upper"],
                "val_performance/log_rank_p":    metrics["log_rank_p"],
                "val_performance/auc_12m":       val_td_auc.get("auc_365d", float("nan")),
                "val_performance/auc_24m":       val_td_auc.get("auc_730d", float("nan")),
                "val_performance/auc_36m":       val_td_auc.get("auc_1095d", float("nan")),
                "val_performance/auc_mean":      val_td_auc["auc_mean"],
            }
            # --auc-days가 기본값(12/24/36m)이 아니면, 실제 계산된 시점들도 원본 day 단위 키로 남긴다.
            for k, v in val_td_auc.items():
                if k.startswith("auc_") and k.endswith("d"):
                    log_dict[f"val_performance/{k}"] = v
            wandb.log(log_dict, step=epoch + 1)

        if score > best_score:
            best_score   = score
            best_metrics = {**metrics, **{f"td_{k}": v for k, v in val_td_auc.items()}, "epoch": epoch + 1}
            epochs_since_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch":            epoch + 1,
                    "val_c_index":      best_score,
                    "val_hr":           metrics["hr"],
                    "val_hr_ci":        (metrics["hr_ci_lower"], metrics["hr_ci_upper"]),
                    "val_log_rank_p":   metrics["log_rank_p"],
                    "val_time_auc":     val_td_auc,
                },
                ckpt_path,
            )
            print(f"  -> checkpoint saved (c_index={best_score:.4f}, HR={metrics['hr']:.3f}, "
                  f"log-rank p={metrics['log_rank_p']:.4f}, AUC_mean={val_td_auc['auc_mean']:.4f})")
            if WANDB_AVAILABLE:
                wandb.run.summary["best_val_c_index"]     = best_score
                wandb.run.summary["best_val_hr"]          = metrics["hr"]
                wandb.run.summary["best_val_hr_ci_lower"] = metrics["hr_ci_lower"]
                wandb.run.summary["best_val_hr_ci_upper"] = metrics["hr_ci_upper"]
                wandb.run.summary["best_val_log_rank_p"]  = metrics["log_rank_p"]
                wandb.run.summary["best_val_auc_mean"]    = val_td_auc["auc_mean"]
                wandb.run.summary["best_epoch"]           = epoch + 1
        else:
            epochs_since_improve += 1
            if (args.early_stop_patience is not None
                    and epochs_since_improve >= args.early_stop_patience):
                print(f"  -> early stop: 최근 {epochs_since_improve} epoch 동안 val c_index 갱신 없음 "
                      f"(best epoch {best_metrics.get('epoch', '-')}, best c_index={best_score:.4f})")
                break

    # 2026-07-26: 작은 validation set(31명)에서 best-val 체크포인트 선택 자체가 노이즈에 취약할
    # 수 있다는 가설(seed126: val_c_index가 3시드 중 최고인데 test는 최저) 검증용 — best-val
    # 선택 없이 마지막 epoch 모델을 그대로 평가해 비교한다. 재학습 불필요: 학습 루프 종료 직후
    # (아래 best checkpoint 리로드 전) 메모리 상의 model이 곧 마지막 epoch 가중치다.
    final_train_metrics = evaluate(model, train_eval_loader, cfg, device, amp_ctx, eval_transform, tile_cache=tile_cache,
                                    desc="final train_eval")
    if test_ds is not None:
        final_test_metrics  = evaluate(model, test_loader, cfg, device, amp_ctx, test_ds.transform, desc="final test")
        final_test_td_auc   = compute_time_dependent_auc(
            final_train_metrics["times"], final_train_metrics["events"],
            final_test_metrics["times"], final_test_metrics["events"], final_test_metrics["risks"],
            eval_days=auc_days,
        )
        print("\n=== Internal Test 성능 (마지막 epoch %d 모델, best-val 선택 없음) ===" % cfg.train.epochs)
        print(_log_line("final_test", final_test_metrics, final_test_td_auc))
        if WANDB_AVAILABLE:
            wandb.run.summary["final_epoch_test_c_index"]    = final_test_metrics["c_index"]
            wandb.run.summary["final_epoch_test_hr"]          = final_test_metrics["hr"]
            wandb.run.summary["final_epoch_test_log_rank_p"]  = final_test_metrics["log_rank_p"]
            wandb.run.summary["final_epoch_test_auc_mean"]    = final_test_td_auc["auc_mean"]
        # 2026-08-15: best-val 체크포인트 선택이 작은 val set(31명) 노이즈에 취약하다는 게
        # 반복 확인돼(M3/M4 fold2/3 등), best 대신 마지막 epoch 모델의 예측을 pooling할 수 있게
        # best-checkpoint와 동일한 CSV 포맷으로 별도 저장한다(_FINALEPOCH 접미사로 구분 —
        # scripts/pool_multiseed_kfold_preds.py --model 인자에 이 태그를 그대로 쓰면 됨).
        if args.fold is not None:
            import csv
            pred_dir = Path(__file__).parent / ".logs" / "kfold_preds"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"{args.dataset}_{model_prefix}_FINALEPOCH_seed{cfg.train.seed}_fold{args.fold}of{args.n_folds}.csv"
            with open(pred_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
                for cid, risk, t, e in zip(final_test_metrics["case_ids"], final_test_metrics["risks"],
                                            final_test_metrics["times"], final_test_metrics["events"]):
                    writer.writerow([cid, risk, t, e])
            print(f"  -> final-epoch fold predictions saved: {pred_path}")
    if external_ds is not None:
        final_external_metrics = evaluate(model, external_loader, cfg, device, amp_ctx, external_ds.transform,
                                           tile_cache=external_tile_cache, desc="final external")
        final_external_td_auc  = compute_time_dependent_auc(
            final_train_metrics["times"], final_train_metrics["events"],
            final_external_metrics["times"], final_external_metrics["events"], final_external_metrics["risks"],
            eval_days=auc_days,
        )
        print(f"=== External Test 성능 ({external_dataset} 전체 코호트, 마지막 epoch 모델) ===")
        print(_log_line("final_external", final_external_metrics, final_external_td_auc))
        if WANDB_AVAILABLE:
            wandb.run.summary["final_epoch_external_c_index"]   = final_external_metrics["c_index"]
            wandb.run.summary["final_epoch_external_hr"]         = final_external_metrics["hr"]
            wandb.run.summary["final_epoch_external_log_rank_p"] = final_external_metrics["log_rank_p"]
            wandb.run.summary["final_epoch_external_auc_mean"]   = final_external_td_auc["auc_mean"]

    # [SWA] 평균 모델(swa_model.module)을 별도로 internal/external test에 평가 — 기존 best-val/
    # 마지막-epoch 리포트와 나란히, 세 번째 관점으로만 추가한다(다른 로직에 영향 없음).
    if swa_model is not None:
        swa_module = swa_model.module
        swa_train_metrics = evaluate(swa_module, train_eval_loader, cfg, device, amp_ctx, eval_transform, tile_cache=tile_cache,
                                      desc="swa train_eval")
        if test_ds is not None:
            swa_test_metrics = evaluate(swa_module, test_loader, cfg, device, amp_ctx, test_ds.transform, desc="swa test")
            swa_test_td_auc  = compute_time_dependent_auc(
                swa_train_metrics["times"], swa_train_metrics["events"],
                swa_test_metrics["times"], swa_test_metrics["events"], swa_test_metrics["risks"],
                eval_days=auc_days,
            )
            print("\n=== Internal Test 성능 (SWA 평균 모델, epoch %d~%d 평균) ===" % (swa_start_epoch, cfg.train.epochs))
            print(_log_line("swa_test", swa_test_metrics, swa_test_td_auc))
            if WANDB_AVAILABLE:
                wandb.run.summary["swa_test_c_index"]   = swa_test_metrics["c_index"]
                wandb.run.summary["swa_test_hr"]         = swa_test_metrics["hr"]
                wandb.run.summary["swa_test_log_rank_p"] = swa_test_metrics["log_rank_p"]
                wandb.run.summary["swa_test_auc_mean"]   = swa_test_td_auc["auc_mean"]
        if external_ds is not None:
            swa_external_metrics = evaluate(swa_module, external_loader, cfg, device, amp_ctx, external_ds.transform,
                                             tile_cache=external_tile_cache, desc="swa external")
            swa_external_td_auc  = compute_time_dependent_auc(
                swa_train_metrics["times"], swa_train_metrics["events"],
                swa_external_metrics["times"], swa_external_metrics["events"], swa_external_metrics["risks"],
                eval_days=auc_days,
            )
            print(f"=== External Test 성능 ({external_dataset} 전체 코호트, SWA 평균 모델) ===")
            print(_log_line("swa_external", swa_external_metrics, swa_external_td_auc))
            if WANDB_AVAILABLE:
                wandb.run.summary["swa_external_c_index"]   = swa_external_metrics["c_index"]
                wandb.run.summary["swa_external_hr"]         = swa_external_metrics["hr"]
                wandb.run.summary["swa_external_log_rank_p"] = swa_external_metrics["log_rank_p"]
                wandb.run.summary["swa_external_auc_mean"]   = swa_external_td_auc["auc_mean"]

    # [SWAD] best epoch를 중심으로 val c_index가 (best - --swad-tolerance) 이상인 연속 구간을
    # 좌우로 넓혀가며 찾고(= flat/plateau window), 그 구간의 epoch 가중치만 균등 평균한 뒤
    # internal/external을 딱 한 번 평가한다 — --swa(고정 마지막 N%% 평균)와 달리 overfit-aware.
    if args.swad and swad_epoch_states:
        best_i = max(range(len(swad_epoch_val_c)), key=lambda i: swad_epoch_val_c[i])
        best_val = swad_epoch_val_c[best_i]
        lo = best_i
        while lo - 1 >= 0 and swad_epoch_val_c[lo - 1] >= best_val - args.swad_tolerance:
            lo -= 1
        hi = best_i
        while hi + 1 < len(swad_epoch_val_c) and swad_epoch_val_c[hi + 1] >= best_val - args.swad_tolerance:
            hi += 1
        window_states = swad_epoch_states[lo:hi + 1]
        print(f"\n[SWAD] plateau window: epoch {lo+1}~{hi+1} (best epoch {best_i+1}, val_c={best_val:.4f}, "
              f"{len(window_states)}개 epoch 평균, tolerance={args.swad_tolerance})")
        swad_avg_state = {k: sum(sd[k].float() for sd in window_states) / len(window_states)
                           for k in window_states[0]}
        swad_ckpt_path = ckpt_path.with_name(ckpt_path.stem + "_swadweights.pt")
        torch.save({"model_state_dict": swad_avg_state, "swad_window": (lo + 1, hi + 1),
                    "swad_best_epoch": best_i + 1, "swad_best_val_c": best_val}, swad_ckpt_path)
        print(f"  -> SWAD 평균 가중치 저장: {swad_ckpt_path}")
        swad_module = copy.deepcopy(model)
        swad_module.load_state_dict(swad_avg_state)
        swad_train_metrics = evaluate(swad_module, train_eval_loader, cfg, device, amp_ctx, eval_transform, tile_cache=tile_cache,
                                       desc="swad train_eval")
        if test_ds is not None:
            swad_test_metrics = evaluate(swad_module, test_loader, cfg, device, amp_ctx, test_ds.transform, desc="swad test")
            swad_test_td_auc  = compute_time_dependent_auc(
                swad_train_metrics["times"], swad_train_metrics["events"],
                swad_test_metrics["times"], swad_test_metrics["events"], swad_test_metrics["risks"],
                eval_days=auc_days,
            )
            print("=== Internal Test 성능 (SWAD 평균 모델, epoch %d~%d 평균) ===" % (lo + 1, hi + 1))
            print(_log_line("swad_test", swad_test_metrics, swad_test_td_auc))
            if WANDB_AVAILABLE:
                wandb.run.summary["swad_test_c_index"]   = swad_test_metrics["c_index"]
                wandb.run.summary["swad_test_hr"]         = swad_test_metrics["hr"]
                wandb.run.summary["swad_test_log_rank_p"] = swad_test_metrics["log_rank_p"]
                wandb.run.summary["swad_test_auc_mean"]   = swad_test_td_auc["auc_mean"]
        if external_ds is not None:
            swad_external_metrics = evaluate(swad_module, external_loader, cfg, device, amp_ctx, external_ds.transform,
                                              tile_cache=external_tile_cache, desc="swad external")
            swad_external_td_auc  = compute_time_dependent_auc(
                swad_train_metrics["times"], swad_train_metrics["events"],
                swad_external_metrics["times"], swad_external_metrics["events"], swad_external_metrics["risks"],
                eval_days=auc_days,
            )
            print(f"=== External Test 성능 ({external_dataset} 전체 코호트, SWAD 평균 모델) ===")
            print(_log_line("swad_external", swad_external_metrics, swad_external_td_auc))
            if WANDB_AVAILABLE:
                wandb.run.summary["swad_external_c_index"]   = swad_external_metrics["c_index"]
                wandb.run.summary["swad_external_hr"]         = swad_external_metrics["hr"]
                wandb.run.summary["swad_external_log_rank_p"] = swad_external_metrics["log_rank_p"]
                wandb.run.summary["swad_external_auc_mean"]   = swad_external_td_auc["auc_mean"]
        del swad_module, window_states, swad_avg_state

    # 학습 종료 후, best checkpoint로 held-out test set을 "딱 한 번" 평가한다.
    # --full-train은 val 자체가 없어 best-val 체크포인트가 존재하지 않으므로, 위에서 이미 계산해둔
    # 마지막 epoch 결과(final_train_metrics)를 그대로 재사용하고 model도 리로드하지 않는다 —
    # 아래 external 평가가 자연히 "마지막 epoch 모델" 기준으로 정확히 1회 수행된다.
    if not args.full_train:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        train_metrics_final = evaluate(model, train_eval_loader, cfg, device, amp_ctx, eval_transform, tile_cache=tile_cache,
                                        desc="train_eval")
        test_metrics = evaluate(model, test_loader, cfg, device, amp_ctx, test_ds.transform, desc="internal test")
        test_td_auc  = compute_time_dependent_auc(
            train_metrics_final["times"], train_metrics_final["events"],
            test_metrics["times"], test_metrics["events"], test_metrics["risks"],
            eval_days=auc_days,
        )
        print("\n=== Internal Test 성능 (같은 코호트 held-out, best checkpoint, epoch %d) ===" % ckpt["epoch"])
        print(_log_line("test", test_metrics, test_td_auc))

        if args.fold is not None:
            import csv
            pred_dir = Path(__file__).parent / ".logs" / "kfold_preds"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"{args.dataset}_{model_prefix}_seed{cfg.train.seed}_fold{args.fold}of{args.n_folds}.csv"
            with open(pred_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
                for cid, risk, t, e in zip(test_metrics["case_ids"], test_metrics["risks"],
                                            test_metrics["times"], test_metrics["events"]):
                    writer.writerow([cid, risk, t, e])
            print(f"  -> fold predictions saved: {pred_path}")

        if WANDB_AVAILABLE:
            wandb.run.summary["test_c_index"]     = test_metrics["c_index"]
            wandb.run.summary["test_hr"]          = test_metrics["hr"]
            wandb.run.summary["test_hr_ci_lower"] = test_metrics["hr_ci_lower"]
            wandb.run.summary["test_hr_ci_upper"] = test_metrics["hr_ci_upper"]
            wandb.run.summary["test_log_rank_p"]  = test_metrics["log_rank_p"]
            wandb.run.summary["test_auc_mean"]    = test_td_auc["auc_mean"]
            wandb.finish()  # [ExternalTest] external은 별도 run(XM 접두)으로 로깅하므로 여기서 main run을 닫는다
    else:
        train_metrics_final = final_train_metrics
        if WANDB_AVAILABLE:
            wandb.finish()

    # [ExternalTest] 학습에 전혀 쓰이지 않은 다른 코호트 전체를 best checkpoint로 딱 한 번 평가한다.
    # censoring 분포(time-dependent AUC)는 internal test와 동일하게 학습 코호트(train split) 기준.
    # wandb는 학습에 쓰인 데이터셋(args.dataset)을 prefix로 유지하되, 모델 구분자에 X를 붙인
    # 별도 run(예: TCGA_XM2_0715::1430)으로 남겨 internal(main) run과 구분한다.
    external_metrics, external_td_auc = None, None
    if external_ds is not None:
        external_metrics = evaluate(model, external_loader, cfg, device, amp_ctx, external_ds.transform,
                                     tile_cache=external_tile_cache, desc="external")
        external_td_auc  = compute_time_dependent_auc(
            train_metrics_final["times"], train_metrics_final["events"],
            external_metrics["times"], external_metrics["events"], external_metrics["risks"],
            eval_days=auc_days,
        )
        ckpt_desc = "마지막 epoch 모델, full-train" if args.full_train else "best checkpoint"
        print(f"\n=== External Test 성능 ({external_dataset} 전체 코호트, {ckpt_desc}) ===")
        print(_log_line("external", external_metrics, external_td_auc))
        # 2026-09-05: k-fold 모드(--fold/--n-folds, --full-train 아님)에서는 external 평가
        # 결과가 화면/wandb에만 남고 CSV로 저장이 안 되던 gap — internal kfold_preds 저장(위
        # 3459행 부근)과 대칭으로 여기서도 저장한다. pool_multiseed_external_preds.py가 찾는
        # 파일명 규칙과 정확히 맞춰야 한다(scripts/experiment_*.py들이 이미 쓰던 관례와 동일).
        if not args.full_train and args.fold is not None:
            import csv
            pred_dir = Path(__file__).parent / ".logs" / "external_preds"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"{external_dataset}_{model_prefix}_seed{cfg.train.seed}_fold{args.fold}of{args.n_folds}.csv"
            with open(pred_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
                for cid, risk, t, e in zip(external_metrics["case_ids"], external_metrics["risks"],
                                            external_metrics["times"], external_metrics["events"]):
                    writer.writerow([cid, risk, t, e])
            print(f"  -> external predictions saved: {pred_path}")
        if args.full_train:
            # 2026-09-02: --full-train은 checkpoint를 저장하지 않아(val이 없어 "best" 선택이
            # 불가능, torch.save가 위 3045행처럼 항상 val 분기 안에서만 호출됨) --eval-external-ckpt
            # 로 나중에 CSV를 다시 뽑을 방법이 없다 — train_light.py --full-train에 적용한 것과
            # 동일하게 이 실행 안에서 바로 저장한다(scripts/pool_fulltrain_external_preds.py 입력용,
            # fold 개념 없이 seed로만 구분).
            import csv
            pred_dir = Path(__file__).parent / ".logs" / "external_preds"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"{external_dataset}_{model_prefix}_FULLTRAIN_seed{cfg.train.seed}.csv"
            with open(pred_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
                for cid, risk, t, e in zip(external_metrics["case_ids"], external_metrics["risks"],
                                            external_metrics["times"], external_metrics["events"]):
                    writer.writerow([cid, risk, t, e])
            print(f"  -> external predictions saved: {pred_path}")
        if WANDB_AVAILABLE:
            external_run_name = f"{args.dataset.upper()}_X{model_prefix}_seed{cfg.train.seed}_{run_ts}"
            wandb.init(
                project="Path-ViT",
                name=external_run_name,
                group=wandb_group,
                config={
                    "dataset":          args.dataset,
                    "external_dataset": external_dataset,
                    "model":            ("ViT_M4" if args.M4
                                          else "ViT_M4A" if args.M4A
                                          else "ViT_MCAT" if args.MCAT
                                          else "ViT_PORPOISE" if args.PORPOISE
                                          else "ViT_M4B" if args.M4B
                                          else "ViT_PM4" if args.PM4
                                          else "ViT_PMA" if args.PMA
                                          else "ViT_M4A_FF" if args.M4A_FF
                                          else "ViT_M2_FF" if args.M2_FF
                                          else "ViT_PMA_FF" if args.PMA_FF
                                          else "ClinicalOnly" if args.M5
                                          else "RNAOnly" if args.M6
                                          else "RNAOnlyExtend" if args.M6X
                                          else "ViT_M2" if args.M2
                                          else "LateFusionViT" if args.fusion
                                          else "ViT_M1_AvgPool" if args.avgpool else "ViT_M1"),
                },
            )
            # wandb.log()로 history를 한 줄 남겨야 Charts에 값이 찍힌다 — summary만 채우면
            # (예전 방식) 그 run의 History가 비어 있어 Charts에는 아무것도 안 보이고
            # Overview의 summary 표에만 값이 존재하는 것처럼 보였다.
            external_log_dict = {
                "external/c_index":     external_metrics["c_index"],
                "external/hr":          external_metrics["hr"],
                "external/hr_ci_lower": external_metrics["hr_ci_lower"],
                "external/hr_ci_upper": external_metrics["hr_ci_upper"],
                "external/log_rank_p":  external_metrics["log_rank_p"],
                "external/auc_12m":     external_td_auc.get("auc_365d", float("nan")),
                "external/auc_24m":     external_td_auc.get("auc_730d", float("nan")),
                "external/auc_36m":     external_td_auc.get("auc_1095d", float("nan")),
                "external/auc_mean":    external_td_auc["auc_mean"],
            }
            for k, v in external_td_auc.items():
                if k.startswith("auc_") and k.endswith("d"):
                    external_log_dict[f"external/{k}"] = v
            wandb.log(external_log_dict)
            wandb.run.summary["external_dataset"] = external_dataset
            wandb.finish()

    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    external_line = (
        f"> External({external_dataset.upper()}) C-index: *{external_metrics['c_index']:.4f}* | "
        f"HR: {external_metrics['hr']:.3f} [{external_metrics['hr_ci_lower']:.3f}, "
        f"{external_metrics['hr_ci_upper']:.3f}] | log-rank p: {external_metrics['log_rank_p']:.4f} | "
        f"AUC(12/24/36m): {external_td_auc.get('auc_365d', float('nan')):.3f}/"
        f"{external_td_auc.get('auc_730d', float('nan')):.3f}/"
        f"{external_td_auc.get('auc_1095d', float('nan')):.3f}\n"
        if external_metrics is not None else ""
    )
    if args.full_train:
        send_slack(
            f":white_check_mark: *Path-ViT ({args.dataset.upper()} OS, --full-train) 학습 완료*\n"
            f"> Epochs: {cfg.train.epochs} (val/internal test 없음 — 코호트 전체를 train으로 사용)\n"
            f"{external_line}"
            f"> 소요 시간: {h}h {m}m {s}s"
        )
    else:
        send_slack(
            f":white_check_mark: *Path-ViT ({args.dataset.upper()} OS) 학습 완료*\n"
            f"> Epochs: {cfg.train.epochs} (best={best_metrics.get('epoch', '-')}) | "
            f"Best val C-index: *{best_score:.4f}* | HR: {best_metrics.get('hr', float('nan')):.3f}\n"
            f"> Internal Test C-index: *{test_metrics['c_index']:.4f}* | HR: {test_metrics['hr']:.3f} "
            f"[{test_metrics['hr_ci_lower']:.3f}, {test_metrics['hr_ci_upper']:.3f}] | "
            f"log-rank p: {test_metrics['log_rank_p']:.4f} | AUC(12/24/36m): "
            f"{test_td_auc.get('auc_365d', float('nan')):.3f}/{test_td_auc.get('auc_730d', float('nan')):.3f}/"
            f"{test_td_auc.get('auc_1095d', float('nan')):.3f}\n"
            f"{external_line}"
            f"> 소요 시간: {h}h {m}m {s}s"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        load_env()
        send_slack(f":x: *Path-ViT (OS) 학습 에러*\n```{type(e).__name__}: {e}```")
        raise
