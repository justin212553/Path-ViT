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
    resolve_tcga_only_rna_genes, pathway_category_gene_ids,
)
from data.patch_utils import (
    FEATURES_AUG_FILENAME, PATCH_TRANSFORM, PATCH_TRANSFORM_AUGMENTED,
    PATCH_TRANSFORM_AUGMENTED_CACHED, PATCH_TRANSFORM_512, build_tile_cache,
)
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


def _stage_ord_from_patient(patient_slides, device) -> dict[str, torch.Tensor] | None:
    """patient_slides[0]에 STAGE_FIELDS(with_staging=True로 로드된 경우만)가 있으면 device로 옮겨
    {field: () 스칼라 long} dict로 반환한다 - "미상"은 -1(data/dataset.py 규약). with_staging=False로
    로드된 데이터셋(--clinical-staging/--stage-aux-weight 둘 다 미사용)이면 None."""
    p = patient_slides[0]
    if STAGE_FIELDS[0] not in p:
        return None
    return {f: p[f].to(device, non_blocking=True) for f in STAGE_FIELDS}


def _patient_risk(
    model, patient_slides, device, amp_ctx, transform, chunk_size, patch_keep_frac: float = 1.0,
    shuffle_patches: bool = False, tile_cache: dict | None = None,
    patch_subsample_generator: torch.Generator | None = None,
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
        slide_spatial_feats = []
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

        patient_embed = torch.stack(slide_embeds).mean(dim=0)      # (D,) — 슬라이드 평균 풀링
        patient_spatial_feat = torch.stack(slide_spatial_feats).mean(dim=0) if slide_spatial_feats else None

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
            # --no-clinical(ViT_PMA use_clinical=False)이면 clinical_encoder 자체가 없다 —
            # getattr로 존재 여부부터 확인해야 한다(2026-07-29, --no-clinical 첫 실사용 중 발견).
            _clinical_enc = getattr(model, "clinical_encoder", None)
            stage_ord = (
                _stage_ord_from_patient(patient_slides, device)
                if _clinical_enc is not None and _clinical_enc.use_staging else None
            )
            patient_embed = model.combine_with_clinical_rna(
                patient_embed, age_years, sex_idx, z_rna, stage_ord=stage_ord,
                spatial_feat=patient_spatial_feat,
            )  # (3D,) (+ spatial_feat_dim, models/spatial_features.py 켜졌을 때만)
        elif hasattr(model, "combine_with_clinical"):
            # --M2_FF: rna_encoder는 있지만(FFN 직전 FiLM용) 최종 결합엔 RNA를 직접 노출하지 않는
            # 모델이라, encoder 존재 여부가 아니라 결합 메서드 존재 여부로 분기해야 한다.
            age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
            sex_idx   = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
            stage_ord = _stage_ord_from_patient(patient_slides, device) if model.clinical_encoder.use_staging else None
            patient_embed = model.combine_with_clinical(
                patient_embed, age_years, sex_idx, stage_ord=stage_ord, spatial_feat=patient_spatial_feat,
            )  # (2D,) (+ spatial_feat_dim, 2026-07-30 — M1/M2에도 dispersion 확장, train_multi.py)
        elif patient_spatial_feat is not None:
            # 2026-07-30: M1(ViT_M1)은 combine 메서드가 없어 patient_embed에 직접 이어붙인다 —
            # use_attn_dispersion=True인 M1을 이전엔 아무도 안 써서 드러나지 않았던 버그
            # (train_multi.py에서 M1/M2에도 dispersion을 확장하며 발견, RuntimeError:
            # LayerNorm 차원 불일치).
            patient_embed = torch.cat([patient_embed, patient_spatial_feat], dim=-1)

        risk = model.risk_head(patient_embed.unsqueeze(0)).view(1)  # (1,)
    return risk, aux_loss, stage_aux_loss


def train_one_epoch(
    model, loader, optimizer, cfg, device, amp_ctx, transform,
    patch_keep_frac: float = 1.0, rna_aux_weight: float = 0.0, stage_aux_weight: float = 0.0,
    shuffle_patches: bool = False, tile_cache: dict | None = None,
    patch_subsample_generator: torch.Generator | None = None,
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
    is_sam = isinstance(optimizer, SAM)

    def _compute_loss(risk_list, time_t, event_t, aux_list, stage_aux_list):
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
        return loss

    def _refresh_batch(patients):
        """SAM 2nd pass용 — perturb된 가중치로 같은 배치(환자 리스트)를 다시 forward한다."""
        risks2, aux2, stage_aux2 = [], [], []
        for ps in patients:
            r2, a2, s2 = _patient_risk(
                model, ps, device, amp_ctx, transform, chunk_size, patch_keep_frac,
                shuffle_patches=shuffle_patches, tile_cache=tile_cache,
                patch_subsample_generator=patch_subsample_generator,
            )
            risks2.append(r2)
            if a2 is not None:
                aux2.append(a2)
            if s2 is not None:
                stage_aux2.append(s2)
        return risks2, aux2, stage_aux2

    def _flush():
        nonlocal risks, times, events, aux_losses, stage_aux_losses, patient_batch, total_loss, total_batches
        if not risks:
            return
        time_t  = torch.cat(times).to(device)
        event_t = torch.cat(events).to(device)

        loss = _compute_loss(risks, time_t, event_t, aux_losses, stage_aux_losses)
        optimizer.zero_grad()
        loss.backward()

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
            risks2, aux2, stage_aux2 = _refresh_batch(patient_batch)
            loss2 = _compute_loss(risks2, time_t, event_t, aux2, stage_aux2)
            loss2.backward()
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

    # mininterval=30: data/patch_utils.py::build_tile_cache와 동일한 이유(비-TTY 로그 폭주 방지).
    for patient_slides in tqdm(loader, desc=desc, unit="patient", mininterval=30):  # 환자 1명 분량의 슬라이드 리스트
        if len(patient_slides) == 0:
            continue
        risk, aux_loss, stage_aux_loss = _patient_risk(
            model, patient_slides, device, amp_ctx, transform, chunk_size, patch_keep_frac,
            shuffle_patches=shuffle_patches, tile_cache=tile_cache,
            patch_subsample_generator=patch_subsample_generator,
        )

        risks.append(risk)
        times.append(patient_slides[0]["OS_time"])
        events.append(patient_slides[0]["OS_event"])
        if aux_loss is not None:
            aux_losses.append(aux_loss)
        if stage_aux_loss is not None:
            stage_aux_losses.append(stage_aux_loss)
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
            "literature_fdr0.1_cptac_only",
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
        "--backbone", type=str, default="resnet50", choices=["resnet50", "uni", "resnet50_norm"],
        help="frozen tile encoder 선택 (기본: resnet50=Lunit SwAV, 2048-dim). uni는 UNI ViT-L/16"
             "(1024-dim, 224 리사이즈) — 미리 `python -m utils.extract_features --backbone uni`로 "
             "features_uni.pt를 뽑아둬야 한다(HuggingFace gated repo 접근 승인 + .env HF_TOKEN 필요). "
             "resnet50_norm은 Macenko stain-normalized 후 같은 ResNet50/Lunit SwAV로 재추출한 "
             "feature(features_norm.pt, utils/extract_features_stain_norm.py) — 인코더 자체는 "
             "resnet50과 동일(2048-dim), 캐싱 파일만 다르다.",
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
        "--exclude-normal-slides", action="store_true",
        help="확인된 정상 조직 슬라이드만 제외하고 케이스당 나머지는 전부 그대로 둔다"
             "(data/dataset.py::_exclude_normal_slides, findings_backlog.md 14번 항목) — "
             "--one-slide-per-case보다 훨씬 덜 급진적인 절충안(TCGA 평균 슬라이드/case "
             "2.52→2.28, CPTAC 3.22→2.76). 기본은 미사용. 켜면 wandb/checkpoint에 _NONORMAL "
             "접미사가 자동으로 붙는다.",
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
    if (args.rna_dim is not None or args.clinical_dim is not None) and not args.PMA:
        raise ValueError("--rna-dim/--clinical-dim은 --PMA에서만 사용 가능합니다.")
    if args.rna_gate_only and not args.PMA:
        raise ValueError("--rna-gate-only는 --PMA에서만 사용 가능합니다.")
    if args.no_clinical and not args.PMA:
        raise ValueError("--no-clinical은 --PMA에서만 사용 가능합니다.")
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
        elif args.rna_genes.endswith("_tcga_only") or args.rna_genes.endswith("_cptac_only"):
            # 이 분기를 먼저 안 걸러 아래 일반 분기로 흘려보내면 leaky한 both-결합
            # literature_guided_gene_ids(N)이 조용히 로드된다 — 반드시 여기서
            # single-cohort/FDR 로더로만 보낸다(data/dataset.py::resolve_tcga_only_rna_genes).
            rna_gene_ids  = resolve_tcga_only_rna_genes(args.rna_genes)
            rna_input_dim = len(rna_gene_ids)
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
        # _AUG = 학습 시 타일 augmentation(seed 고정, 1회성 features_aug.pt) 사용 표시.
        model_prefix += "_AUG"
    if args.dropout is not None and args.dropout != 0.3:
        # _DROP{dropout} = cfg.model.dropout(기본 0.3) 스윕 표시.
        model_prefix += f"_DROP{args.dropout:g}"
    if args.rna_dim is not None:
        model_prefix += f"_RNADIM{args.rna_dim}"
    if args.clinical_dim is not None:
        model_prefix += f"_CLINDIM{args.clinical_dim}"
    if args.embed_dim is not None:
        model_prefix += f"_EMBDIM{args.embed_dim}"
    if args.rna_gate_only:
        model_prefix += "_RNAGATE"
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
    if args.pretrained_wsi_trunk:
        model_prefix += "_PRETRAINED"
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
        fold=args.fold, n_folds=args.n_folds,
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
            PATCH_TRANSFORM_AUGMENTED_CACHED if (args.tile_augment and not cfg.data.precomputed) else None
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
    if args.init_seed is not None:
        torch.manual_seed(args.init_seed)
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
                         precomputed=cfg.data.precomputed, backbone=args.backbone,
                         rna_dim=args.rna_dim, clinical_dim=args.clinical_dim,
                         rna_gate_only=args.rna_gate_only, use_clinical=not args.no_clinical,
                         **stage_kwargs).to(device)
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
                        precomputed=cfg.data.precomputed, backbone=args.backbone,
                        use_attn_dispersion=args.attn_dispersion, **stage_kwargs).to(device)
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
                        use_attn_dispersion=args.attn_dispersion).to(device)
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

    if args.sam:
        optimizer = SAM(
            filter(lambda p: p.requires_grad, model.parameters()),
            torch.optim.AdamW, rho=args.sam_rho,
            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        )
    else:
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

    best_score   = -1.0
    best_metrics = {}
    for epoch in range(cfg.train.epochs):
        lr_now        = optimizer.param_groups[0]["lr"]
        loss          = train_one_epoch(model, train_loader, optimizer, cfg, device, amp_ctx, train_ds.transform,
                                         patch_keep_frac=args.patch_keep_frac, rna_aux_weight=args.rna_aux_weight,
                                         stage_aux_weight=args.stage_aux_weight,
                                         shuffle_patches=args.shuffle_patches, tile_cache=tile_cache,
                                         patch_subsample_generator=patch_subsample_generator,
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
