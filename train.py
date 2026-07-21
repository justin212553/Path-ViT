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
import math
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import Config
from data.dataset import (
    WSISurvivalDataset, CLINICAL_PATHS, pdac_subtype_gene_ids, literature_guided_gene_ids,
    pathway_category_gene_ids,
)
from data.patch_utils import PATCH_TRANSFORM_AUGMENTED
from models import (
    ViT_M1, ViT_M1_AvgPool, LateFusionViT, ViT_M2, ViT_M4, ViT_M4A, ViT_M4B,
    ViT_PM4, ViT_PMA, ViT_M4A_FF, ViT_M2_FF, ViT_PMA_FF, ClinicalOnly, RNAOnly, RNAOnlyExtend,
)
from models.rna_predictor import RNAPredictionHead
from models.stage_predictor import StagePredictionHead
from models.clinical_encoder import age_stats_from_csv, STAGE_FIELDS, stage_stats_from_df
from data.fit_clusters import CENTROIDS_DIR
from utils import load_env, send_slack
from utils.losses import cox_ph_loss
from utils.metrics import compute_survival_metrics, compute_time_dependent_auc


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


def _stage_ord_from_patient(patient_slides, device) -> dict[str, torch.Tensor] | None:
    """patient_slides[0]에 STAGE_FIELDS(with_staging=True로 로드된 경우만)가 있으면 device로 옮겨
    {field: () 스칼라 long} dict로 반환한다 - "미상"은 -1(data/dataset.py 규약). with_staging=False로
    로드된 데이터셋(--clinical-staging/--stage-aux-weight 둘 다 미사용)이면 None."""
    p = patient_slides[0]
    if STAGE_FIELDS[0] not in p:
        return None
    return {f: p[f].to(device, non_blocking=True) for f in STAGE_FIELDS}


def _patient_risk(
    model, patient_slides, device, amp_ctx, transform, chunk_size, patch_keep_frac: float = 1.0
):
    """환자 1명이 보유한 슬라이드 전부를 forward해 임베딩을 평균 풀링한 뒤 risk score(scalar)를 계산한다.
    Returns: (risk, aux_loss, stage_aux_loss) — aux_loss는 model.rna_aux_head가 있을 때만 텐서
    (--rna-aux-weight, models/rna_predictor.py 참조), stage_aux_loss는 model.stage_aux_head가
    있을 때만 텐서(--stage-aux-weight, models/stage_predictor.py 참조), 둘 다 없으면 None.

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
            if getattr(model, "clinical_encoder", None) is not None and model.clinical_encoder.use_staging:
                stage_kwargs["stage_ord"] = _stage_ord_from_patient(patient_slides, device)
            return model(age_years, sex_idx, **stage_kwargs), None, None

    with amp_ctx:
        z_rna = None
        rna_true = None
        if hasattr(model, "rna_encoder"):
            rna = patient_slides[0]["rna"].to(device, non_blocking=True)
            z_rna = model.encode_rna(rna)  # (D,)
            rna_true = rna

        slide_embeds = []
        slide_meanpool_embeds = []
        for slide in patient_slides:
            coords = slide["coords"]
            features = slide.get("features")
            patch_paths = slide.get("patch_paths")

            if model.training and patch_keep_frac < 1.0:
                n = coords.shape[0]
                k = max(1, round(n * patch_keep_frac))
                if k < n:
                    idx = torch.randperm(n)[:k]
                    coords = coords[idx]
                    if features is not None:
                        features = features[idx]
                    if patch_paths is not None:
                        patch_paths = [patch_paths[i] for i in idx.tolist()]

            coords = coords.to(device, non_blocking=True)
            forward_kwargs = {"rna_context": z_rna} if z_rna is not None else {}
            if features is not None:
                out = model(coords, features=features, **forward_kwargs)
            else:
                out = model(coords, patch_paths=patch_paths,
                             transform=transform, chunk_size=chunk_size, **forward_kwargs)
            slide_embeds.append(out["embed"])
            if "meanpool_embed" in out:
                slide_meanpool_embeds.append(out["meanpool_embed"])

        patient_embed = torch.stack(slide_embeds).mean(dim=0)      # (D,) — 슬라이드 평균 풀링

        patient_meanpool = None
        if slide_meanpool_embeds and (hasattr(model, "rna_aux_head") or hasattr(model, "stage_aux_head")):
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

        if hasattr(model, "combine_with_clinical_rna"):
            age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
            sex_idx   = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
            stage_ord = _stage_ord_from_patient(patient_slides, device) if model.clinical_encoder.use_staging else None
            patient_embed = model.combine_with_clinical_rna(
                patient_embed, age_years, sex_idx, z_rna, stage_ord=stage_ord
            )  # (3D,)
        elif hasattr(model, "combine_with_clinical"):
            # --M2_FF: rna_encoder는 있지만(FFN 직전 FiLM용) 최종 결합엔 RNA를 직접 노출하지 않는
            # 모델이라, encoder 존재 여부가 아니라 결합 메서드 존재 여부로 분기해야 한다.
            age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
            sex_idx   = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
            stage_ord = _stage_ord_from_patient(patient_slides, device) if model.clinical_encoder.use_staging else None
            patient_embed = model.combine_with_clinical(
                patient_embed, age_years, sex_idx, stage_ord=stage_ord
            )  # (2D,)

        risk = model.risk_head(patient_embed.unsqueeze(0)).view(1)  # (1,)
    return risk, aux_loss, stage_aux_loss


def train_one_epoch(
    model, loader, optimizer, cfg, device, amp_ctx, transform,
    patch_keep_frac: float = 1.0, rna_aux_weight: float = 0.0, stage_aux_weight: float = 0.0,
) -> float:
    model.train()
    if hasattr(model, "cnn") and model.cnn.backbone is not None:
        model.cnn.backbone.eval()  # frozen backbone의 BN을 population stats(eval)로 고정 — train/eval 분포 불일치 방지
    total_loss    = 0.0
    total_batches = 0
    chunk_size    = cfg.train.cnn_chunk_size
    batch_size    = cfg.train.cox_batch_size

    risks, times, events, aux_losses, stage_aux_losses = [], [], [], [], []

    def _flush():
        nonlocal risks, times, events, aux_losses, stage_aux_losses, total_loss, total_batches
        if not risks:
            return
        risk_t  = torch.cat(risks)
        time_t  = torch.cat(times).to(device)
        event_t = torch.cat(events).to(device)

        loss = cox_ph_loss(risk_t, time_t, event_t)
        if rna_aux_weight > 0 and aux_losses:
            # --rna-aux-weight(models/rna_predictor.py): WSI 표현이 RNA 발현도 예측하도록
            # 보조 loss를 더한다 — 생존 라벨(환자당 1개, censoring으로 더 약함)만으로
            # 62만 파라미터짜리 WSI 브랜치를 학습시키는 게 병목이라는 진단(model_zoo.md)에 대한
            # 대응. 결합 방식이 아니라 학습 신호 자체를 보강한다.
            loss = loss + rna_aux_weight * torch.stack(aux_losses).mean()
        if stage_aux_weight > 0 and stage_aux_losses:
            # --stage-aux-weight(models/stage_predictor.py): 위와 동일 원리, 타깃만 T-stage/grade.
            loss = loss + stage_aux_weight * torch.stack(stage_aux_losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item()
        total_batches += 1
        risks, times, events, aux_losses, stage_aux_losses = [], [], [], [], []

    for patient_slides in loader:                # 환자 1명 분량의 슬라이드 리스트
        if len(patient_slides) == 0:
            continue
        risk, aux_loss, stage_aux_loss = _patient_risk(
            model, patient_slides, device, amp_ctx, transform, chunk_size, patch_keep_frac
        )

        risks.append(risk)
        times.append(patient_slides[0]["OS_time"])
        events.append(patient_slides[0]["OS_event"])
        if aux_loss is not None:
            aux_losses.append(aux_loss)
        if stage_aux_loss is not None:
            stage_aux_losses.append(stage_aux_loss)

        if len(risks) >= batch_size:
            _flush()

    _flush()  # 마지막 남은 partial batch

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, cfg, device, amp_ctx, transform) -> dict:
    model.eval()
    all_risks, all_times, all_events = [], [], []
    chunk_size = cfg.train.cnn_chunk_size

    for patient_slides in loader:
        if len(patient_slides) == 0:
            continue
        risk, _, _ = _patient_risk(model, patient_slides, device, amp_ctx, transform, chunk_size)

        all_risks.append(risk.float().item())
        all_times.append(float(patient_slides[0]["OS_time"].item()))
        all_events.append(int(patient_slides[0]["OS_event"].item()))

    risks  = np.array(all_risks)
    times  = np.array(all_times)
    events = np.array(all_events)
    return {
        **compute_survival_metrics(risks, times, events),
        "risks": risks, "times": times, "events": events,
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
        "--group-ts", type=str, default=None,
        help="wandb Group 이름(<모델종류>_<group-ts>)에 쓸 타임스탬프(MMDD::HHMM 형식). "
             "여러 시드/코호트를 스윕하는 래퍼 스크립트가 첫 실행 전에 한 번 계산해 모든 "
             "python train.py 호출에 동일한 값을 넘기면, 그 세션에서 나온 같은 모델 종류의 "
             "모든 run(internal+external 전부)이 wandb에서 하나의 Group으로 묶인다. "
             "생략하면 이 실행 자체의 시작 시각을 써서 이 run 하나만의 그룹이 된다.",
    )
    parser.add_argument(
        "--rna-genes", type=str, default="subtype",
        choices=["subtype", "literature_1000", "literature_1500", "literature_2000", "pathway8"],
        help="RNA 브랜치(--M4/--M4A/--M4B/--PM4/--PMA/--M6/--M6X) 입력 유전자셋 선택. "
             "subtype(기본): pdac_subtype_gene_ids(), Bailey/Moffitt subtype 분류용 ~340개. "
             "literature_{1000,1500,2000}: data/select_rnaseq_genes.py 산출물 — 문헌 큐레이션 "
             "PDAC 유전자를 train split 내부 Cox score test 순위로 우선 배치하고 나머지를 "
             "Cox 순위로 채운, 생존 예측에 직접 최적화된 유전자셋(레퍼런스 방법론 이식). "
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
        "--rna-aux-weight", type=float, default=0.0,
        help="WSI 표현이 RNA 발현도 예측하도록 하는 보조과제(auxiliary task) 가중치, "
             "models/rna_predictor.py::RNAPredictionHead. 0.0(기본)이면 비활성. RNA를 쓰는 "
             "모델(--M4/--M4A/--M4B/--PM4/--PMA)에서만 적용되며(rna_encoder 필요), attn_pool의 "
             "RNA 개입과 무관하게 ViT 직후 mean-pooled 표현(RNA-free)에서 예측한다 - HE2RNA류 "
             "설계. cox loss에 이 가중치를 곱해 더한다. 0.0 초과면 wandb/checkpoint에 _AUX "
             "접미사가 자동으로 붙는다.",
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
        "--image", action="store_true",
        help="패치 jpg/png를 매 forward마다 ResNet50으로 직접 인코딩 (기본: data/extract_features.py로 "
             "사전 추출한 features.pt 사용)",
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet50", choices=["resnet50", "uni", "resnet50_norm"],
        help="frozen tile encoder 선택 (기본: resnet50=Lunit SwAV, 2048-dim). uni는 UNI ViT-L/16"
             "(1024-dim, 224 리사이즈) — 미리 `python -m utils.extract_features --backbone uni`로 "
             "features_uni.pt를 뽑아둬야 한다(HuggingFace gated repo 접근 승인 + .env HF_TOKEN 필요). "
             "resnet50_norm은 Macenko stain-normalized 후 같은 ResNet50/Lunit SwAV로 재추출한 "
             "feature(features_norm.pt, utils/extract_features_stain_norm.py) — 인코더 자체는 "
             "resnet50과 동일(2048-dim), 캐싱 파일만 다르다.",
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
        "--dropout", type=float, default=None,
        help="cfg.model.dropout(기본 0.3) 덮어쓰기 — ViT/Nystromformer, ABMIL, RNA/Clinical "
             "인코더 전체가 공유하는 dropout rate 스윕용(findings_backlog.md 13번 항목 후속, "
             "risk head 자체 dropout과는 별개 실험). 기본값(None)과 다르면 wandb/checkpoint에 "
             "_DROP{dropout} 접미사가 자동으로 붙는다.",
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
        help="레퍼런스 M4_Train.ipynb::get_train_cached_patch_transform()과 동일하게 학습 시 "
             "타일에 RandomHorizontalFlip/VerticalFlip/ColorJitter/GaussianBlur를 실시간으로 "
             "적용한다(data/patch_utils.py::PATCH_TRANSFORM_AUGMENTED) — 매 epoch 매 forward마다 "
             "raw 이미지를 다시 backbone에 태우므로 --image(비-precomputed 모드)와 함께만 "
             "쓸 수 있다(features.pt 캐시를 쓰는 기본 모드에서는 애초에 backbone을 다시 안 태우니 "
             "augmentation이 적용될 지점이 없다). val/test/external은 항상 증강 없는 기본 "
             "PATCH_TRANSFORM을 쓴다(레퍼런스도 eval엔 미적용). --image 없이 켜면 에러. "
             "주의: 학습이 극도로 느려진다(로컬 GPU 기준 ResNet50 1024px ~24 img/s — findings_"
             "backlog.md 참조). 켜면 wandb/checkpoint에 _AUG 접미사가 자동으로 붙는다.",
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
    return parser.parse_args()


def _log_line(prefix: str, metrics: dict, td_auc: dict | None = None) -> str:
    """print용 한 줄 로그 문자열 (c_index/HR/log-rank p [+ time-dependent AUC])."""
    line = (
        f"{prefix}_c_index={metrics['c_index']:.4f} | {prefix}_HR={metrics['hr']:.3f} "
        f"[{metrics['hr_ci_lower']:.3f}, {metrics['hr_ci_upper']:.3f}] | "
        f"{prefix}_logrank_p={metrics['log_rank_p']:.4f}"
    )
    if td_auc is not None:
        line += f" | {prefix}_AUC_mean={td_auc['auc_mean']:.4f}"
    return line


def main():
    load_env()
    args   = _parse_args()
    cfg    = Config()
    cfg.data.precomputed = not args.image
    if args.seed is not None:
        cfg.data.seed  = args.seed
        cfg.train.seed = args.seed
    if args.patches_root_tcga is not None:
        cfg.data.patches_root_tcga = args.patches_root_tcga
    if args.patches_root_cptac is not None:
        cfg.data.patches_root_cptac = args.patches_root_cptac
    if args.dropout is not None:
        cfg.model.dropout = args.dropout

    # [LateFusion] --fusion 플래그 시 cluster_centroids.pt 로드 검증
    if args.fusion and not cfg.data.precomputed:
        raise ValueError("--fusion은 precomputed(features.pt) 모드에서만 지원됩니다. --image와 함께 사용 불가.")
    if args.M2 and args.fusion:
        raise ValueError("--M2(Clinical fusion)와 --fusion(Cluster fusion)은 동시에 지원되지 않습니다.")
    if (args.M4 or args.M4A or args.M4B or args.PM4 or args.PMA or args.M4A_FF or args.M2_FF or args.PMA_FF) and args.fusion:
        raise ValueError("--M4/--M4A/--M4B/--PM4/--PMA(Clinical+RNA fusion)와 --fusion(Cluster fusion)은 동시에 지원되지 않습니다.")
    if (args.M5 or args.M6 or args.M6X) and args.fusion:
        raise ValueError("--M5/--M6/--M6X(WSI-free)와 --fusion(Cluster fusion, WSI 전제)은 동시에 지원되지 않습니다.")
    if args.avgpool and (args.M2 or args.M4 or args.M4A or args.M4B or args.PM4 or args.PMA or args.M4A_FF or args.M2_FF or args.PMA_FF or args.M5 or args.M6 or args.M6X or args.fusion):
        raise ValueError(
            "--avgpool은 --M1(기본)에서만 지원됩니다 — "
            "--M2/--M4/--M4A/--M4B/--PM4/--PMA/--M5/--M6/--M6X/--fusion과 동시 사용 불가."
        )
    if args.clinical_staging and not (
        args.M2 or args.M4 or args.M4A or args.M4B or args.PM4 or args.PMA
        or args.M4A_FF or args.M2_FF or args.PMA_FF or args.M5
    ):
        raise ValueError(
            "--clinical-staging은 ClinicalEncoder를 쓰는 모델(--M2/--M4/--M4A/--M4B/--PM4/"
            "--PMA/--M4A_FF/--M2_FF/--M5)에서만 사용 가능합니다."
        )
    if args.fusion and args.backbone != "resnet50":
        raise ValueError(
            "--fusion(LateFusionViT)의 cluster_centroids.pt는 ResNet50 raw feature(2048-dim) "
            "기준으로 사전 계산돼 있어 --backbone uni(1024-dim)와 호환되지 않습니다. "
            "uni로 --fusion을 쓰려면 data/fit_clusters.py를 features_uni.pt 기준으로 다시 돌려야 합니다."
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

    # [Clinical] --M2/--M4/--M4A/--M4B/--PM4/--PMA/--M5 시 age z-score 정규화 통계를 학습 코호트
    # (args.dataset)에서 계산해 고정한다(extract_rna_clinical.py의 "데이터셋 내부 z-score
    # 정규화" 관례와 동일). dataset="both"면 두 코호트 clinical.csv를 합쳐 통계를 계산한다.
    if args.M2 or args.M4 or args.M4A or args.M4B or args.PM4 or args.PMA or args.M4A_FF or args.M2_FF or args.PMA_FF or args.M5:
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

    # [RNA] --M4/--M4A/--M4B/--PM4/--PMA/--M6/--M6X 시 RNAEncoder 입력 유전자셋을 --rna-genes로
    # 고른다 — 기본(subtype)은 Bailey/Moffitt subtype 분류용 ~340개, literature_{1000,1500,2000}은
    # data/select_rnaseq_genes.py 산출물(생존 예측에 직접 최적화된 유전자셋). WSISurvivalDataset에
    # 그대로 넘겨 실제 로드되는 컬럼과 rna_input_dim이 항상 일치하게 한다.
    uses_rna = args.M4 or args.M4A or args.M4B or args.PM4 or args.PMA or args.M4A_FF or args.M2_FF or args.PMA_FF or args.M6 or args.M6X
    rna_pathway_categories = None
    if uses_rna:
        if args.rna_genes == "pathway8":
            rna_pathway_categories = pathway_category_gene_ids()
            rna_gene_ids  = None
            rna_input_dim = len(rna_pathway_categories)
        else:
            rna_gene_ids = (
                pdac_subtype_gene_ids() if args.rna_genes == "subtype"
                else literature_guided_gene_ids(int(args.rna_genes.split("_")[1]))
            )
            rna_input_dim = len(rna_gene_ids)
    else:
        rna_gene_ids = None
        rna_input_dim = None

    set_seed(cfg.train.seed)
    start_time = datetime.now()
    device = torch.device(cfg.train.device)

    torch.backends.cudnn.benchmark = True

    amp_ctx = _make_amp_ctx()

    if args.M4:
        model_prefix = "M4"
    elif args.M4A:
        model_prefix = "M4A"
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
    elif args.rna_genes != "subtype":
        # _EX = literature_guided_gene_ids() 등 확장 유전자셋(레퍼런스 방식) 사용 표시.
        # wandb에서 기본(subtype, ~340개) run과 섞이지 않게 이름/그룹에 항상 붙인다.
        model_prefix += "_EX"
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
    if args.exclude_normal_slides:
        # _NONORMAL = 확인된 정상 조직 슬라이드만 제외(케이스당 나머지는 전부 유지) 표시.
        model_prefix += "_NONORMAL"
    if args.one_slide_per_case:
        # _1SLIDE = 케이스당 대표 슬라이드 1장만 사용 표시(findings_backlog.md 14번 항목).
        model_prefix += "_1SLIDE"
    if args.tile_augment:
        if not args.image:
            raise ValueError("--tile-augment는 --image(비-precomputed 모드)와 함께만 쓸 수 있습니다.")
        # _AUG = 학습 시 실시간 타일 augmentation 사용 표시.
        model_prefix += "_AUG"
    if args.dropout is not None and args.dropout != 0.3:
        # _DROP{dropout} = cfg.model.dropout(기본 0.3) 스윕 표시.
        model_prefix += f"_DROP{args.dropout:g}"

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
                "dataset":               args.dataset,
                "external_dataset":      external_dataset,
            },
        )

    with_clinical = args.M2 or args.M4 or args.M4A or args.M4B or args.PM4 or args.PMA or args.M4A_FF or args.M2_FF or args.PMA_FF or args.M5
    with_rna = args.M4 or args.M4A or args.M4B or args.PM4 or args.PMA or args.M4A_FF or args.M2_FF or args.PMA_FF or args.M6 or args.M6X
    ds_kwargs = dict(
        with_clinical=with_clinical, with_staging=with_staging, with_rna=with_rna,
        feature_backbone=args.backbone,
        rna_gene_ids=rna_gene_ids, rna_pathway_categories=rna_pathway_categories,
        one_slide_per_case=args.one_slide_per_case,
        exclude_normal_slides=args.exclude_normal_slides,
    )
    # --tile-augment는 학습 split에서만 적용한다(val/test/external은 항상 증강 없는 기본 transform).
    train_ds = WSISurvivalDataset(
        cfg.data, dataset=args.dataset, split="train",
        transform=PATCH_TRANSFORM_AUGMENTED if args.tile_augment else None,
        **ds_kwargs,
    )
    val_ds   = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="val",   **ds_kwargs)
    test_ds  = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="test",  **ds_kwargs)
    # [ExternalTest] 학습에 전혀 쓰이지 않은 코호트 전체(split="all") — 없으면 None
    external_ds = (
        WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", **ds_kwargs)
        if external_dataset else None
    )

    dl_kwargs = dict(
        batch_size=1,
        collate_fn=_identity_collate,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.data.num_workers > 0),
        prefetch_factor=2 if cfg.data.num_workers > 0 else None,
    )
    train_loader      = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **dl_kwargs)
    val_loader        = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader       = DataLoader(test_ds,  shuffle=False, **dl_kwargs)
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
    if args.M4:
        model = ViT_M4(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                        precomputed=cfg.data.precomputed, backbone=args.backbone, **stage_kwargs).to(device)
    elif args.M4A:
        model = ViT_M4A(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                         precomputed=cfg.data.precomputed, backbone=args.backbone, **stage_kwargs).to(device)
    elif args.M4B:
        model = ViT_M4B(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                         precomputed=cfg.data.precomputed, backbone=args.backbone, **stage_kwargs).to(device)
    elif args.PM4:
        model = ViT_PM4(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                         precomputed=cfg.data.precomputed, backbone=args.backbone, **stage_kwargs).to(device)
    elif args.PMA:
        model = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                         precomputed=cfg.data.precomputed, backbone=args.backbone, **stage_kwargs).to(device)
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
        model = ClinicalOnly(cfg.model, age_mean=age_mean, age_std=age_std, **stage_kwargs).to(device)
    elif args.M6:
        model = RNAOnly(cfg.model, rna_input_dim=rna_input_dim).to(device)
    elif args.M6X:
        model = RNAOnlyExtend(cfg.model, rna_input_dim=rna_input_dim).to(device)
    elif args.M2:
        model = ViT_M2(cfg.model, age_mean=age_mean, age_std=age_std,
                        precomputed=cfg.data.precomputed, backbone=args.backbone, **stage_kwargs).to(device)
    elif args.fusion:
        model = LateFusionViT(cfg.model, cluster_centroids, precomputed=cfg.data.precomputed).to(device)
    elif args.avgpool:
        model = ViT_M1_AvgPool(cfg.model, precomputed=cfg.data.precomputed, backbone=args.backbone).to(device)
    else:
        model = ViT_M1(cfg.model, precomputed=cfg.data.precomputed, backbone=args.backbone).to(device)
    if hasattr(model, "cnn") and model.cnn.backbone is not None:
        model.cnn.backbone.requires_grad_(False)

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

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg)

    mode = "precomputed features" if cfg.data.precomputed else "raw image (--image)"
    print(f"Mode: {mode}")
    # [Clinical/RNA/LateFusion] 모델 종류 출력
    if args.M4:
        print(f"Model: ViT_M4 (ViT+ABMIL(RNA-guided FiLM) + Clinical age/sex MLP + RNA-seq MLP, "
              f"age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.M4A:
        print(f"Model: ViT_M4A (ViT+CoAttentionPooling(RNA query) + Clinical age/sex MLP + RNA-seq MLP, "
              f"age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.M4B:
        print(f"Model: ViT_M4B (ViT+pre-ViT FiLM(RNA) token conditioning + Clinical age/sex MLP + "
              f"RNA-seq MLP, age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.PM4:
        print(f"Model: ViT_PM4 (ViT+다성분 pooling(mean/std/attn/top-k) + RNA post-hoc gate + "
              f"Clinical age/sex MLP, age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.PMA:
        print(f"Model: ViT_PMA (ViT+다성분 pooling + CoAttention(RNA query, 4개 관점) + "
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
        f"| cnn_chunk={cfg.train.cnn_chunk_size} | workers={cfg.data.num_workers}"
    )
    ckpt_dir  = Path(__file__).parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # [Clinical/RNA/LateFusion] 모델 종류·backbone별로 별도 checkpoint 저장 — ablation 결과 보존.
    # backbone을 태그에 안 넣으면 --backbone uni/resnet50을 오가며 돌릴 때 같은 파일을 덮어써서
    # 서로 다른 feature 차원의 checkpoint가 섞여버린다.
    tag = args.dataset if args.backbone == "resnet50" else f"{args.dataset}_{args.backbone}"
    if args.rna_genes == "pathway8":
        tag += "_PW8"
    elif args.rna_genes != "subtype":
        # gene set이 다르면 같은 모델 종류라도 입력 차원이 달라 checkpoint가 호환되지 않는다 —
        # backbone 태그와 같은 이유로 파일명에 반드시 구분자를 남긴다.
        tag += "_EX"
    if args.patch_keep_frac < 1.0:
        tag += "_SS"
    if args.rna_aux_weight > 0:
        tag += "_AUX"
    if args.stage_aux_weight > 0:
        tag += "_AUX2"
    if args.clinical_staging:
        tag += "_STG"
    if args.M4:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_clinical_rna.pt"
    elif args.M4A:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_clinical_rna_coattn.pt"
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
    elif args.M2:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_clinical.pt"
    elif args.fusion:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_fusion.pt"
    elif args.avgpool:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_avgpool.pt"
    else:
        ckpt_path = ckpt_dir / f"survival_{tag}_best.pt"

    best_score   = -1.0
    best_metrics = {}
    for epoch in range(cfg.train.epochs):
        lr_now        = optimizer.param_groups[0]["lr"]
        loss          = train_one_epoch(model, train_loader, optimizer, cfg, device, amp_ctx, train_ds.transform,
                                         patch_keep_frac=args.patch_keep_frac, rna_aux_weight=args.rna_aux_weight,
                                         stage_aux_weight=args.stage_aux_weight)
        train_metrics = evaluate(model, train_eval_loader, cfg, device, amp_ctx, train_ds.transform)
        metrics       = evaluate(model, val_loader, cfg, device, amp_ctx, val_ds.transform)
        val_td_auc    = compute_time_dependent_auc(
            train_metrics["times"], train_metrics["events"],
            metrics["times"], metrics["events"], metrics["risks"],
        )
        scheduler.step()

        c_index = metrics.get("c_index", float("nan"))
        score   = c_index if not math.isnan(c_index) else -1.0
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
                "val_performance/auc_12m":       val_td_auc["auc_365d"],
                "val_performance/auc_24m":       val_td_auc["auc_730d"],
                "val_performance/auc_36m":       val_td_auc["auc_1095d"],
                "val_performance/auc_mean":      val_td_auc["auc_mean"],
            }
            wandb.log(log_dict, step=epoch + 1)

        if score > best_score:
            best_score   = score
            best_metrics = {**metrics, **{f"td_{k}": v for k, v in val_td_auc.items()}, "epoch": epoch + 1}
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

    # 학습 종료 후, best checkpoint로 held-out test set을 "딱 한 번" 평가한다.
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    train_metrics_final = evaluate(model, train_eval_loader, cfg, device, amp_ctx, train_ds.transform)
    test_metrics = evaluate(model, test_loader, cfg, device, amp_ctx, test_ds.transform)
    test_td_auc  = compute_time_dependent_auc(
        train_metrics_final["times"], train_metrics_final["events"],
        test_metrics["times"], test_metrics["events"], test_metrics["risks"],
    )
    print("\n=== Internal Test 성능 (같은 코호트 held-out, best checkpoint, epoch %d) ===" % ckpt["epoch"])
    print(_log_line("test", test_metrics, test_td_auc))
    if WANDB_AVAILABLE:
        wandb.run.summary["test_c_index"]     = test_metrics["c_index"]
        wandb.run.summary["test_hr"]          = test_metrics["hr"]
        wandb.run.summary["test_hr_ci_lower"] = test_metrics["hr_ci_lower"]
        wandb.run.summary["test_hr_ci_upper"] = test_metrics["hr_ci_upper"]
        wandb.run.summary["test_log_rank_p"]  = test_metrics["log_rank_p"]
        wandb.run.summary["test_auc_mean"]    = test_td_auc["auc_mean"]
        wandb.finish()  # [ExternalTest] external은 별도 run(XM 접두)으로 로깅하므로 여기서 main run을 닫는다

    # [ExternalTest] 학습에 전혀 쓰이지 않은 다른 코호트 전체를 best checkpoint로 딱 한 번 평가한다.
    # censoring 분포(time-dependent AUC)는 internal test와 동일하게 학습 코호트(train split) 기준.
    # wandb는 학습에 쓰인 데이터셋(args.dataset)을 prefix로 유지하되, 모델 구분자에 X를 붙인
    # 별도 run(예: TCGA_XM2_0715::1430)으로 남겨 internal(main) run과 구분한다.
    external_metrics, external_td_auc = None, None
    if external_ds is not None:
        external_metrics = evaluate(model, external_loader, cfg, device, amp_ctx, external_ds.transform)
        external_td_auc  = compute_time_dependent_auc(
            train_metrics_final["times"], train_metrics_final["events"],
            external_metrics["times"], external_metrics["events"], external_metrics["risks"],
        )
        print(f"\n=== External Test 성능 ({external_dataset} 전체 코호트, best checkpoint) ===")
        print(_log_line("external", external_metrics, external_td_auc))
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
            wandb.log({
                "external/c_index":     external_metrics["c_index"],
                "external/hr":          external_metrics["hr"],
                "external/hr_ci_lower": external_metrics["hr_ci_lower"],
                "external/hr_ci_upper": external_metrics["hr_ci_upper"],
                "external/log_rank_p":  external_metrics["log_rank_p"],
                "external/auc_12m":     external_td_auc["auc_365d"],
                "external/auc_24m":     external_td_auc["auc_730d"],
                "external/auc_36m":     external_td_auc["auc_1095d"],
                "external/auc_mean":    external_td_auc["auc_mean"],
            })
            wandb.run.summary["external_dataset"] = external_dataset
            wandb.finish()

    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    external_line = (
        f"> External({external_dataset.upper()}) C-index: *{external_metrics['c_index']:.4f}* | "
        f"HR: {external_metrics['hr']:.3f} [{external_metrics['hr_ci_lower']:.3f}, "
        f"{external_metrics['hr_ci_upper']:.3f}] | log-rank p: {external_metrics['log_rank_p']:.4f} | "
        f"AUC(12/24/36m): {external_td_auc['auc_365d']:.3f}/{external_td_auc['auc_730d']:.3f}/"
        f"{external_td_auc['auc_1095d']:.3f}\n"
        if external_metrics is not None else ""
    )
    send_slack(
        f":white_check_mark: *Path-ViT ({args.dataset.upper()} OS) 학습 완료*\n"
        f"> Epochs: {cfg.train.epochs} (best={best_metrics.get('epoch', '-')}) | "
        f"Best val C-index: *{best_score:.4f}* | HR: {best_metrics.get('hr', float('nan')):.3f}\n"
        f"> Internal Test C-index: *{test_metrics['c_index']:.4f}* | HR: {test_metrics['hr']:.3f} "
        f"[{test_metrics['hr_ci_lower']:.3f}, {test_metrics['hr_ci_upper']:.3f}] | "
        f"log-rank p: {test_metrics['log_rank_p']:.4f} | AUC(12/24/36m): "
        f"{test_td_auc['auc_365d']:.3f}/{test_td_auc['auc_730d']:.3f}/{test_td_auc['auc_1095d']:.3f}\n"
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
