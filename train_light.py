"""
TCGA-PAAD / CPTAC-PDA OS 예측 학습 스크립트 — WSI 없이 Clinical/RNA만 쓰는 모델 전용
(--M5 ClinicalOnly, --M6 RNAOnly, --M6X RNAOnlyExtend, --M7 ClinicalRNAOnly).

train.py와의 관계: train.py는 CNN/ViT/ABMIL을 포함한 전체 파이프라인(M1/M2/M4/M4A/M4B)과
WSI-free 모델(M5/M6/M6X)까지 한 스크립트에서 처리한다. 이 스크립트는 WSI 처리가 전혀
필요 없는 모델만 따로 떼어, 패치 forward/CNN/ViT/ABMIL/AMP 없이 훨씬 가볍고 빠르게 돈다.

가장 중요한 차이는 **하이퍼파라미터**다 — train.py(및 config.py::TrainConfig)의 lr=1e-5는
ViT self-attention+ABMIL이 포함된 WSI 스택의 학습 안정성을 위해 낮게 잡은 값인데,
M5/M6/M6X를 train.py에 배선하면서 이 값을 그대로 물려받았다(실수). 이 스크립트는
config.py::LightTrainConfig(lr=1e-3, Adam 기본값 수준)를 쓴다 — 예전 독립 스크립트였던
train_clinical_rna_only.py(M7)가 이미 이 lr로 지금까지 가장 좋은 external 성능(0.575)을
냈던 값과 같다. 모델 폭(embed_dim/dropout)은 train.py와 동일하게 cfg.model(ModelConfig)을
그대로 쓴다 — 그래야 train.py 배선 결과와 이 스크립트 결과의 차이가 "아키텍처"가 아니라
"학습 설정(lr/schedule)" 때문임을 깨끗하게 분리해서 볼 수 있다.

사용법:
    python train_light.py --dataset tcga --seed 42 --M6 --external
    python train_light.py --dataset both --seed 42 --M7
"""
import argparse
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
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
    resolve_tcga_only_rna_genes, literature_guided_gene_ids_single_cohort,
    literature_guided_gene_ids_intersection, pathway_category_gene_ids, pathway_flat_gene_ids,
)
from models import ClinicalOnly, RNAOnly, RNAOnlyExtend, ClinicalRNAOnly, HDP
from models.clinical_encoder import (
    age_stats_from_csv, margin_stats_from_csv, margin_stats_from_df, stage_stats_from_df, STAGE_FIELDS,
    mutation_stats_from_csv, mutation_stats_from_df, MUTATION_FIELDS,
)
from utils import load_env, send_slack
from utils.losses import cox_ph_loss
from utils.metrics import compute_survival_metrics, compute_time_dependent_auc


# --HDP(비지도 k-means 군집, 라벨 없음)와 --HDP-PRETRAIN(PanNuke 라벨로 학습된 진짜 종양
# 함량 회귀 head, 2026-09-01 사용자 결정으로 추가) 두 소스를 같은 HDP 모델 클래스(hist_dim만
# 다름)로 돌린다 — 어느 소스든 "환자 단위 저차원 WSI feature 벡터"라는 형태는 동일.
HDP_FEATURE_SOURCES = {
    "cluster": {
        "paths": {
            "tcga": Path("data/cluster_features_uni2native_tcga.csv"),
            "cptac": Path("data/cluster_features_uni2native_cptac.csv"),
        },
        # data/compute_cluster_features_uni2native.py 산출 — prop=비율, disp=공간 dispersion,
        # intravar=군집 내부 임베딩 분산(분화도 heterogeneity 대체 통계, "분화도"라 부르지 않음),
        # centdist=전역 군집 중심까지 거리(성숙도 대체 통계, "성숙도"라 부르지 않음).
        "prefixes": ("prop_", "disp_", "intravar_", "centdist_"),
    },
    "pretrain": {
        "paths": {
            "tcga": Path("data/tumor_content_uni2native_tcga.csv"),
            "cptac": Path("data/tumor_content_uni2native_cptac.csv"),
        },
        # scripts/apply_hdp_pretrain_head.py 산출 — PanNuke(핵 단위 라벨) 기반으로 학습된
        # 회귀 head를 적용한, 진짜 의미가 있는 단일 축(종양 함량)의 4개 통계. "클래스"가
        # 군집 10개가 아니라 1개(종양)라 4차원으로 끝난다(K=1인 셈).
        "prefixes": ("mean_tumor_content", "tumor_heterogeneity", "tumor_dispersion", "frac_high_tumor", "content_entropy"),
    },
    "pretrain_resmatch": {
        "paths": {
            "tcga": Path("data/tumor_content_uni2native_resmatch_tcga.csv"),
            "cptac": Path("data/tumor_content_uni2native_resmatch_cptac.csv"),
        },
        # 2026-09-02: scripts/apply_hdp_pretrain_head.py --head-path ..._resmatch.pt --out-tag
        # _resmatch 산출 — PanNuke 학습 이미지를 uni2native와 같은 배율로 보이게 보정한 head
        # (val_corr 0.60->0.78, findings_backlog.md) 적용판. "pretrain"과 컬럼 구성은 동일.
        # h5(DX 슬라이드만) 기준이라 TCGA 2명/CPTAC 15명은 여전히 fallback.
        "prefixes": ("mean_tumor_content", "tumor_heterogeneity", "tumor_dispersion", "frac_high_tumor", "content_entropy"),
    },
    "pretrain_resmatch_full": {
        "paths": {
            "tcga": Path("data/tumor_content_uni2native_resmatch_full_tcga.csv"),
            "cptac": Path("data/tumor_content_uni2native_resmatch_full_cptac.csv"),
        },
        # 2026-09-02: apply_hdp_pretrain_head.py --source tiletree(HPC uni2native tile 트리 전체,
        # DX+TS+BS 전 타입) + resmatch head 조합 — TCGA는 152/152 완전 커버(fallback 없음),
        # CPTAC은 tile 자체가 없는 15명만 여전히 fallback(h5/tiletree 문제 아님, 더 근본적 공백).
        "prefixes": ("mean_tumor_content", "tumor_heterogeneity", "tumor_dispersion", "frac_high_tumor", "content_entropy"),
    },
}


class ClusterHistLookup:
    """case_id -> (4*K,) 군집 feature 텐서(z-score 정규화 완료). 커버 안 되는 case_id는 전체
    평균(정규화 후 0벡터와 동일)으로 fallback하고 처음 조회될 때 1번만 경고를 찍는다."""

    def __init__(self, lookup: dict, fallback: torch.Tensor):
        self.lookup = lookup
        self.fallback = fallback
        self._warned: set[str] = set()

    def __getitem__(self, case_id: str) -> torch.Tensor:
        if case_id in self.lookup:
            return self.lookup[case_id]
        if case_id not in self._warned:
            print(f"  [경고] case_id={case_id}: uni2native cluster feature 없음 -> 전체 평균(0벡터)으로 대체")
            self._warned.add(case_id)
        return self.fallback


def _load_cluster_histograms(datasets: list[str], source: str = "cluster") -> tuple["ClusterHistLookup", int]:
    """HDP_FEATURE_SOURCES[source] 산출 CSV를 로드해 {case_id: (D,) z-score 정규화 tensor}
    딕셔너리로 합친다. --HDP(source="cluster")/--HDP-PRETRAIN(source="pretrain") 공용.

    비율/공간 스케일(px)/임베딩 분산/임베딩 거리(또는 pretrain의 평균/분산/dispersion/비율)는
    단위가 전부 달라 정규화 없이 하나의 선형층(models/hdp.py::HDP.hist_linear)에 넣으면 스케일
    큰 항이 사실상 gradient를 독점한다 — clinical cox_add(age/margin/stage)와 동일 관례로 각
    열을 전체 코호트 기준 z-score한다(cluster centroid/pretrain head 둘 다 fold 구분 없이
    전체 코호트로 fit·적용했으므로 이 정도의 "전체 통계 사용"은 새로운 leakage가 아니라 기존
    파이프라인과 일관된 선택).

    공식 UNI2-h feature(data/uni2h_official_features/)가 생존 코호트의 일부 환자를 못 커버한다
    (2026-09-01 확인: TCGA 152명 중 2명, CPTAC 159명 중 15명 누락 — MahmoodLab 공식 배포 자체가
    이 환자들 슬라이드를 포함 안 함). 코호트 크기를 그대로 유지해야 M1~M7 결과표와 비교 가능하므로
    (환자를 빼면 fold/코호트가 달라짐), 이런 환자는 버리지 않고 전체 평균(정규화 후 0벡터)으로
    대체한다 — "이 환자에 대해선 군집/종양함량 정보를 모른다"는 뜻의 중립적 fallback.

    Returns: (case_id -> (D,) float tensor 딕셔너리, D)
    """
    import pandas as pd
    spec = HDP_FEATURE_SOURCES[source]
    frames = [pd.read_csv(spec["paths"][d]) for d in datasets]
    df = pd.concat(frames, ignore_index=True)
    feat_cols = sorted(c for c in df.columns if any(c.startswith(p) for p in spec["prefixes"]))
    mean = df[feat_cols].mean()
    std = df[feat_cols].std(ddof=0).replace(0.0, 1.0)
    df_z = (df[feat_cols] - mean) / std
    lookup = {cid: torch.tensor(row.astype("float32"))
              for cid, row in zip(df["case_id"], df_z.to_numpy())}
    fallback = torch.zeros(len(feat_cols), dtype=torch.float32)  # 정규화 후 코호트 평균 = 0벡터
    return ClusterHistLookup(lookup, fallback), len(feat_cols)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# train.py::_BRANCH_ATTRS/_branch_param_groups와 동일 관례(2026-09-02, --clinical-lr-mult 이식) —
# top-level 속성 이름으로 clinical/rna/other 3그룹으로 나눈다. models/clinical_rna_only.py::
# ClinicalRNAOnly(cox_add)와 models/hdp.py::HDP 둘 다 이 속성명(clinical_linear/rna_encoder)을
# 그대로 쓰므로 별도 매핑 없이 재사용 가능.
_BRANCH_ATTRS = {
    "clinical": ("clinical_encoder", "clinical_linear"),
    "rna": ("rna_encoder", "rna_linear"),
}


def _branch_param_groups(model) -> dict[str, list]:
    groups: dict[str, list] = {"clinical": [], "rna": [], "other": []}
    attr_to_key = {attr: key for key, attrs in _BRANCH_ATTRS.items() for attr in attrs}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        top = name.split(".")[0]
        groups[attr_to_key.get(top, "other")].append(p)
    return groups


def _build_scheduler(optimizer, cfg):
    total  = cfg.light.epochs
    warmup = cfg.light.warmup_epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _identity_collate(batch: list) -> list:
    return batch[0]


def _patient_risk(model, patient_slides, device, cluster_hist_lookup: dict | None = None) -> torch.Tensor:
    """WSI 없이 환자 단위 메타데이터(age/sex 및/또는 rna, --HDP면 군집 히스토그램까지)만으로
    risk score를 계산한다.

    model이 HDP(models/hdp.py)면 age/sex/rna/cluster_hist 전부, ClinicalRNAOnly(M7)면
    age/sex/rna, rna_encoder만 있으면 RNAOnly/RNAOnlyExtend(M6/M6X), 그 외(clinical_encoder만)는
    ClinicalOnly(M5) — forward 시그니처가 모델마다 다르므로 분기한다. 2026-07-29:
    ClinicalRNAOnly의 combine_mode="film"/"cox_add"는 clinical_encoder 속성이 없어(그 대신
    film_gamma/clinical_linear) hasattr(model, "clinical_encoder")만으로는 M7을 못 잡는 버그가
    있었다 — isinstance로 명시적으로 잡는다.
    """
    p = patient_slides[0]
    is_hdp = isinstance(model, HDP)
    is_m7 = isinstance(model, ClinicalRNAOnly)
    has_rna = hasattr(model, "rna_encoder")
    if is_hdp:
        hdp_clinical_kwargs = {}
        if getattr(model, "use_margin", False):
            hdp_clinical_kwargs["margin_ord"] = p["margin_ord"].to(device, non_blocking=True)
        if getattr(model, "use_staging", False):
            hdp_clinical_kwargs["stage_ord"] = {f: p[f].to(device, non_blocking=True) for f in STAGE_FIELDS}
        if getattr(model, "use_mutation", False):
            hdp_clinical_kwargs["mutation_ord"] = {f: p[f].to(device, non_blocking=True) for f in MUTATION_FIELDS}
        case_id = p["case_id"]
        cluster_hist = cluster_hist_lookup[case_id].to(device, non_blocking=True)
        return model(
            p["age_years"].to(device, non_blocking=True),
            p["sex_idx"].to(device, non_blocking=True),
            p["rna"].to(device, non_blocking=True),
            cluster_hist,
            **hdp_clinical_kwargs,
        )
    if is_m7:
        m7_clinical_kwargs = {}
        # 2026-08-05: use_margin은 이제 combine_mode 무관하게 model 자체에 있다(film/cox_add도
        # margin 지원) — model.clinical_encoder(concat 전용, film/cox_add엔 없음)가 아니라
        # model.use_margin을 직접 봐야 film/cox_add에서도 margin_ord가 전달된다.
        if getattr(model, "use_margin", False):
            m7_clinical_kwargs["margin_ord"] = p["margin_ord"].to(device, non_blocking=True)
        # 2026-08-06: use_staging도 동일 관례(combine_mode 무관하게 model 자체에 있음).
        if getattr(model, "use_staging", False):
            m7_clinical_kwargs["stage_ord"] = {f: p[f].to(device, non_blocking=True) for f in STAGE_FIELDS}
        if getattr(model, "use_mutation", False):
            m7_clinical_kwargs["mutation_ord"] = {f: p[f].to(device, non_blocking=True) for f in MUTATION_FIELDS}
        return model(
            p["age_years"].to(device, non_blocking=True),
            p["sex_idx"].to(device, non_blocking=True),
            p["rna"].to(device, non_blocking=True),
            **m7_clinical_kwargs,
        )
    if has_rna:
        return model(p["rna"].to(device, non_blocking=True))
    # 2026-08-21: ClinicalOnly(M5)의 raw_linear=True 버전은 clinical_encoder 자체가 없다
    # (models/clinical_only.py) — M7과 동일하게 model 자체의 use_margin/use_staging을 본다.
    margin_kwargs = {}
    if getattr(model, "use_margin", False):
        margin_kwargs["margin_ord"] = p["margin_ord"].to(device, non_blocking=True)
    if getattr(model, "use_staging", False):
        margin_kwargs["stage_ord"] = {f: p[f].to(device, non_blocking=True) for f in STAGE_FIELDS}
    if getattr(model, "use_mutation", False):
        margin_kwargs["mutation_ord"] = {f: p[f].to(device, non_blocking=True) for f in MUTATION_FIELDS}
    return model(
        p["age_years"].to(device, non_blocking=True),
        p["sex_idx"].to(device, non_blocking=True),
        **margin_kwargs,
    )


def train_one_epoch(model, loader, optimizer, device, batch_size: int,
                     cluster_hist_lookup: dict | None = None) -> float:
    model.train()
    total_loss, total_batches = 0.0, 0
    risks, times, events = [], [], []

    def _flush():
        nonlocal risks, times, events, total_loss, total_batches
        if not risks:
            return
        loss = cox_ph_loss(torch.cat(risks), torch.cat(times).to(device), torch.cat(events).to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
        risks.clear(); times.clear(); events.clear()

    for patient_slides in loader:
        if len(patient_slides) == 0:
            continue
        risks.append(_patient_risk(model, patient_slides, device, cluster_hist_lookup))
        times.append(patient_slides[0]["OS_time"])
        events.append(patient_slides[0]["OS_event"])
        if len(risks) >= batch_size:
            _flush()
    _flush()
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, cluster_hist_lookup: dict | None = None) -> dict:
    model.eval()
    all_risks, all_times, all_events, all_case_ids = [], [], [], []
    for patient_slides in loader:
        if len(patient_slides) == 0:
            continue
        risk = _patient_risk(model, patient_slides, device, cluster_hist_lookup)
        all_risks.append(risk.float().item())
        all_times.append(float(patient_slides[0]["OS_time"].item()))
        all_events.append(int(patient_slides[0]["OS_event"].item()))
        all_case_ids.append(patient_slides[0]["case_id"])
    risks, times, events = np.array(all_risks), np.array(all_times), np.array(all_events)
    return {
        **compute_survival_metrics(risks, times, events),
        "risks": risks, "times": times, "events": events, "case_ids": all_case_ids,
    }


def _log_line(prefix: str, metrics: dict, td_auc: dict | None = None) -> str:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="cptac", choices=["tcga", "cptac", "both"])
    parser.add_argument("--seed", type=int, default=None,
                         help="cfg.data.seed / cfg.light.seed를 함께 덮어쓴다.")
    parser.add_argument(
        "--init-seed", type=int, default=None,
        help="2026-07-27: train.py --init-seed와 동일 — 모델 가중치 초기화만 --seed와 분리된 값으로 "
             "고정한다(모델 생성 직전 이 값으로 재시딩, 생성 직후 --seed로 복원). M7_EX에서도 "
             "초기화가 seed 간 external 격차의 주 원인인지 확인하기 위한 용도.",
    )
    parser.add_argument(
        "--auc-days", type=str, default="365,730,1095",
        help="2026-07-28: train.py --auc-days와 동일 — time-dependent AUC 계산 시점(day, 콤마 "
             "구분). 기본 12/24/36개월(365,730,1095). 예: --auc-days 91,182,365,730.",
    )
    parser.add_argument(
        "--external", action="store_true",
        help="internal test와 별도로, 학습에 전혀 쓰지 않은 반대 코호트 전체를 external test로 "
             "평가한다(train.py --external과 동일한 의미). --dataset both와는 함께 못 쓴다.",
    )
    parser.add_argument("--group-ts", type=str, default=None,
                         help="wandb Group 타임스탬프. train.py --group-ts와 동일한 관례.")
    parser.add_argument(
        "--fold", type=int, default=None,
        help="주어지면(0-based) internal train/val/test를 단일 6:2:2 대신 K-fold(data/dataset.py::"
             "_kfold_case_split)로 배정한다 — 이 fold를 test로, 나머지를 다시 60:20으로 train/val "
             "배정. fold=0..n_folds-1을 전부 돌려 test 예측을 pool_kfold_preds.py로 이어붙이면 "
             "internal 표본이 코호트 전체 크기가 된다. --fold를 주면 예측을 "
             ".logs/kfold_preds/에 CSV로 저장한다. None(기본)이면 기존 단일 split.",
    )
    parser.add_argument("--n-folds", type=int, default=5, help="--fold와 함께 쓰는 전체 fold 개수.")
    parser.add_argument(
        "--full-train", action="store_true",
        help="2026-08-16: train.py --full-train와 동일 관례 — 6:2:2/k-fold split 없이 코호트 "
             "전체를 train으로 쓴다(val/internal test 자체가 없음). val이 없어 best-val "
             "체크포인트 선택이 불가능하므로 마지막 epoch 모델로 external을 정확히 1회 평가한다 "
             "(--external과 함께 써야 의미가 있다). M1~M4(train.py)에서 k-fold pooled 앙상블과 "
             "거의 동일한 external 결과를 낸 게 확인돼(±0.01 이내), M5/M6/M7에도 동일하게 적용.",
    )
    parser.add_argument(
        "--rna-genes", type=str, default="subtype",
        choices=[
            "subtype", "literature_1000", "literature_1500", "literature_2000",
            "literature_500_tcga_only", "literature_1000_tcga_only",
            "literature_1500_tcga_only", "literature_fdr0.1_tcga_only",
            "literature_fdr0.1_cptac_only", "purist", "purist_top20_tcga_only",
            "literature_1500_intersection", "pathway8",
            "variance_100_tcga_only", "variance_250_tcga_only", "variance_500_tcga_only",
            "variance_1000_tcga_only", "variance_1500_tcga_only", "purist_pathway8",
            "pathway8_flat",
        ],
        help="RNA 브랜치(--M6/--M6X/--M7) 입력 유전자셋 선택. purist: data/compute_purist_subtype.py "
             "산출물(PurIST basal-like 확률, 1차원) — TCGA/CPTAC 어느 쪽에도 fit하지 않는 고정 "
             "계수 분류기라 leakage 원천 차단, gene-selection 자체가 없어 표본 크기(n=91)에 "
             "흔들리지 않음. train.py --rna-genes와 동일한 관례 "
             "(subtype 외 선택 시 wandb/checkpoint에 _EX 접미사 자동 부착). **주의**: literature_"
             "{1000,1500,2000}은 TCGA+CPTAC train split을 Stouffer로 결합해 뽑은 것이라 --dataset "
             "tcga --external처럼 반대 코호트를 external test로 쓰는 실행에는 leakage가 있다 "
             "(findings_backlog.md) — --dataset both 비교에만 쓸 것. literature_{500,1500}_tcga_only, "
             "literature_fdr0.1_tcga_only는 TCGA train split만 사용(CPTAC 미참조)한 leakage-free "
             "버전(뒤의 fdr0.1은 top-N 대신 BH-FDR q<0.1 통계적 컷오프, data/select_rnaseq_genes.py "
             "--fdr-threshold) — --dataset tcga --external 전용(다른 조합이면 에러). "
             "literature_fdr0.1_cptac_only는 반대 방향(CPTAC train split만 사용, --dataset cptac "
             "--external 전용) — CPTAC이 TCGA보다 train event 비율이 훨씬 높아(87명 중 70 events "
             "vs TCGA 91명 중 50) 이 방향이 더 잘 되는지 비교하려는 용도. purist_top20_tcga_only: "
             "purist(1차원, chance 수준이었음)와 literature_500_tcga_only 계열(과적합 경향)의 "
             "절충 — PurIST 확률 1차원 + TCGA train split Cox-score 상위 20개 개별 유전자를 "
             "concat(총 21차원). --dataset tcga --external 전용. literature_1500_intersection: "
             "TCGA-only 순위와 CPTAC-only 순위가 독립적으로 겹치는 유전자만(data/"
             "select_rnaseq_genes.py --intersection) — 실측상 top-1500끼리는 10%도 안 겹쳐서 "
             "top-6000 each까지 내려가야 1500개가 채워짐. 양쪽 라벨을 각자 안에서만 써서 "
             "방향 상관없이(--dataset tcga/cptac 둘 다) leakage 없이 사용 가능.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="cfg.light.lr(기본 1e-3) 덮어쓰기. findings_backlog.md 3번 항목 - lr=1e-3이 M6를 "
             "train_c_index 0.99까지 과적합시키는 게 스모크 테스트로 확인된 바 있어, M7_EX 등 "
             "baseline 수치를 lr=1e-5(WSI 모델과 동일)로 재검증할 때 사용. 기본값(None)과 다르면 "
             "wandb/checkpoint에 _LR{lr} 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=None,
        help="cfg.light.weight_decay(기본 1e-2) 덮어쓰기. 레퍼런스(Leeyoungsup/"
             "pancreatic_cancer_pathology) M7 학습 레시피(weight_decay=1e-3)를 맞출 때 사용. "
             "기본값(None)과 다르면 wandb/checkpoint에 _WD{wd} 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="cfg.light.epochs(기본 30) 덮어쓰기. 레퍼런스 M7 레시피(epochs=100)를 맞출 때 사용.",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Early stopping patience(epoch 수) - val_c_index가 이 횟수만큼 연속으로 최고 기록을 "
             "못 넘으면 조기 종료한다. 기본값(None)이면 비활성(--epochs 끝까지 고정 학습, 기존 동작). "
             "레퍼런스 M7 레시피는 patience=20.",
    )
    parser.add_argument(
        "--match-reference-cohort", action="store_true",
        help="레퍼런스(Leeyoungsup/pancreatic_cancer_pathology) M4/M7의 케이스 포함 기준"
             "(24개월 시점 생존 여부 확정 + WSI 보유, data/reference_cohort.py 참조)으로 "
             "case를 제한한다. 기본은 미사용(우리 기존 cohort 그대로).",
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--M5", action="store_true", help="ClinicalOnly (age/sex만).")
    model_group.add_argument("--M6", action="store_true", help="RNAOnly (RNA-seq만).")
    model_group.add_argument("--M6X", action="store_true", help="RNAOnlyExtend (RNA-seq, G->256->256 인코더).")
    model_group.add_argument("--M7", action="store_true", help="ClinicalRNAOnly (age/sex + RNA-seq).")
    model_group.add_argument(
        "--HDP", action="store_true",
        help="Human Doctor Prognosis(models/hdp.py) — M7(age/sex+RNA-seq)에 WSI 유래 군집 "
             "히스토그램(K차원, data/compute_cluster_histograms_uni2native.py 산출)을 cox_add로 "
             "추가한다. 2026-09-01: attention/MIL로 patch 중요도를 152개 라벨만으로 처음부터 "
             "발견시키는 접근이 이번 세션 내내 실패해(MCAT/PORPOISE/sharpening), '조직 유형에 "
             "사람이 이름 붙이는' 대안도 시도했으나 TCGA/CPTAC 어디에도 PAAD용 PNI/TIL/종양비율 "
             "라벨이 없어 불가능했다 — 그래서 이름 없이 UNI2-h(공식 스펙) frozen feature를 "
             "라벨-프리 k-means로 군집화한 뒤, 환자별 군집 비율 자체를 clinical/RNA와 같은 방식의 "
             "저차원 구조화 feature로 넣고 어느 군집이 hazard와 상관있는지는 생존 라벨이 직접 "
             "결정하게 한다.",
    )
    model_group.add_argument(
        "--HDP-PRETRAIN", action="store_true", dest="HDP_PRETRAIN",
        help="HDP와 같은 모델 클래스(models/hdp.py)지만 WSI feature 소스가 다르다 — 군집이 "
             "아니라, PanNuke(핵 단위 라벨, scripts/train_hdp_pretrain_head.py로 pancreas "
             "subset 195장에서 학습, held-out val_corr=0.60) 기반으로 학습된 진짜 종양 함량 "
             "회귀 head를 우리 코호트에 적용한(scripts/apply_hdp_pretrain_head.py) 4차원 통계 "
             "(평균/heterogeneity/dispersion/고함량비율)를 cox_add로 추가한다. 2026-09-01 "
             "사용자 결정 — --HDP(비지도 군집, 라벨 없음)가 M7 대비 개선을 못 보여서, 진짜 "
             "라벨로 학습시킨 버전을 별도로 검증한다.",
    )
    parser.add_argument(
        "--clinical-margin", action="store_true",
        help="--M5/--M7 공통(M5_R/M7_R, M7은 --combine-mode concat 전용). age/sex에 절제연 "
             "상태(residual_disease: R0=완전절제 < R1 < R2=잔존종양)를 추가한다(data/"
             "extract_rna_clinical.py 2026-08-04 추가 필드). TCGA/CPTAC 각각 단독 Cox 회귀로 "
             "개별 유의(HR≈1.6, p=0.04/0.01) — age(코호트 간 방향 불일치)나 sex(개별 코호트 "
             "비유의)보다 신뢰도 높은 clinical 신호라 M5가 노이즈 수준(null c-index 대비 <1σ)을 "
             "벗어나는지 확인하는 ablation. 켜면 wandb/checkpoint에 _R 접미사가 자동으로 "
             "붙는다(예: M5_R, M7_EXTfdr0.1_R).",
    )
    parser.add_argument(
        "--no-age-sex", action="store_true",
        help="--clinical-margin과 함께 사용(M5_R_ONLY/M7_..._R_ONLY). age/sex를 아예 빼고 margin만 "
             "입력으로 쓴다 — age는 TCGA(HR=1.20/10y)와 CPTAC(HR=0.94/10y)에서 방향조차 안 맞아, margin의 "
             "더 확실한 신호를 age/sex의 노이즈가 희석시키는 건 아닌지 확인하는 ablation. "
             "켜면 model_prefix에 _ONLY가 추가로 붙는다(M5_R_ONLY).",
    )
    parser.add_argument(
        "--clinical-staging", action="store_true",
        help="2026-08-06: train.py --clinical-staging과 동일 — AJCC 병기(T/N/M)+grade를 추가한다"
             "(--M5/--M7). univariate 검정에서 개별 필드는 margin만큼 깨끗하게 양쪽 코호트에서 "
             "유의하진 않았지만(ajcc_n TCGA만 p=0.048, ajcc_m/grade는 CPTAC만 유의), 4개를 함께 "
             "넣으면 PMA+UNI+cox_add에서 external을 M3(WSI+RNA, clinical 없음) 위로 처음 끌어올린 "
             "걸 확인한 뒤 M7/M5도 같은 레시피(margin+staging)로 맞춰 internal/external 괴리를 "
             "비교하려는 용도. 켜면 model_prefix에 _STG 접미사가 자동으로 붙는다.",
    )
    parser.add_argument(
        "--clinical-mutation", action="store_true",
        help="2026-09-02: PDAC 4대 driver gene(KRAS/TP53/SMAD4/CDKN2A) 체성 변이 유무를 추가한다"
             "(data/extract_mutations.py, GDC MAF 기반). WSI 쪽(HDP/PMA 계열)이 branch를 늘려도 "
             "구조적으로 RNA에 못 이기는 걸 반복 확인한 뒤(concat 아키텍처 실험까지), 새 branch "
             "대신 margin/staging과 동일한 (z_score, known_flag) 관례로 clinical에 얹는다. "
             "univariate로는 KRAS가 TCGA에서 c=0.60(이 세션 raw feature 중 최고)이었으나 CPTAC "
             "에서 c=0.52로 재현 안 됨 — 코호트 간 KRAS 변이율 자체가 다름(58.8% vs 81.0%, "
             "variant calling 파이프라인 차이 의심). 켜면 model_prefix에 _MUT 접미사가 붙는다.",
    )
    parser.add_argument(
        "--raw-linear", action="store_true",
        help="--M5 전용, 2026-08-21 — ClinicalEncoder(MLP) 없이 raw z-score feature를 바로 "
             "Linear(1)에 넣는 고전적 Cox 회귀로 M5를 만든다. M7의 clinical cox_add ablation에서 "
             "clinical 신호가 약할 땐 MLP 경유가 raw feature 직결보다 못하다는 결과(internal "
             "-0.025)가 나와서, M5도 MLP 유무 자체가 성능에 영향을 주는지 참고 삼아 확인하는 "
             "ablation(models/clinical_only.py::ClinicalOnly raw_linear 참조). 켜면 model_prefix에 "
             "_RAWLIN 접미사가 붙는다.",
    )
    parser.add_argument(
        "--clinical-dim", type=int, default=None,
        help="--M7 전용: clinical_encoder 출력 차원(기본 None=모듈 상수, models/"
             "clinical_rna_only.py — 2026-08-05부터 64, M1~M6와 균일). 필요하면 이 인자로 "
             "덮어쓸 수 있다(예: 레퍼런스 비대칭 16으로 되돌리는 ablation).",
    )
    parser.add_argument(
        "--rna-dim", type=int, default=None,
        help="--M7 전용: rna_encoder 출력 차원(기본 None=모듈 상수, 2026-08-05부터 64). "
             "--clinical-dim과 같이 덮어쓸 수 있다.",
    )
    parser.add_argument(
        "--clinical-dropout", type=float, default=0.0,
        help="--M7 전용: clinical_encoder MLP 내부 dropout(기본 0.0=기존 동작). 레퍼런스"
             "(tabular_survival.py::ClinicalEmbedding)엔 Dropout(0.25)이 있는데 우리는 없었다는 "
             "차이가 M7의 clinical 브랜치가 다른 모달리티를 누르는 원인인지 검증하는 ablation.",
    )
    parser.add_argument(
        "--sex-onehot", action="store_true",
        help="--M7 전용: sex를 스칼라 이진 인덱스(0/1) 대신 one-hot(2차원, 레퍼런스 clinical_dim=3 "
             "방식)으로 인코딩한다. 표현력은 수학적으로 등가이지만 초기화 분산/Adam 최적화 경로가 "
             "달라져 결과가 갈릴 수 있는지 검증하는 ablation.",
    )
    parser.add_argument(
        "--risk-hidden-dim", type=int, default=None,
        help="--M7 --combine-mode concat 전용: risk_head에 레퍼런스(tabular_survival.py::"
             "ClinicalRNASeqSurvivalModel.classifier) 사양(LayerNorm→Dropout→Linear→GELU→"
             "Dropout→Linear)처럼 히든 레이어를 추가한다. 기본 None이면 기존(히든 없는 "
             "LayerNorm→Linear 직결) 동작 그대로. 13번 항목에서 레퍼런스 절대 폭(hidden=128)은 "
             "negative였는데, 우리 폭(rna=64/clinical=4)에 맞춰 같은 비율로 축소(예: 32)해서 "
             "재검증할 때 사용.",
    )
    parser.add_argument(
        "--risk-dropout", type=float, default=0.0,
        help="--risk-hidden-dim과 함께 사용하는 risk_head 내부 dropout(기본 0.0). 레퍼런스는 0.4.",
    )
    parser.add_argument(
        "--eval-external-ckpt", type=str, default=None,
        help="2026-08-08: train.py --eval-external-ckpt와 동일 관례 — 학습을 전혀 하지 않고, 이 "
             "경로의 checkpoint를 로드해 --external 코호트 전체에 대해서만 딱 한 번 평가한 뒤 "
             "환자별 예측을 .logs/external_preds/에 CSV로 저장하고 즉시 종료한다"
             "(scripts/pool_multiseed_external_preds.py 입력용). 다른 모든 인자(--M5/--M6/--M7/"
             "--rna-genes/--fold 등)는 그 checkpoint를 만들 때와 정확히 똑같이 줘야 한다. "
             "--external 없이 쓰면 에러.",
    )
    parser.add_argument(
        "--combine-mode", type=str, default="concat", choices=["concat", "film", "cox_add"],
        help="--M7 전용: clinical과 RNA를 합치는 방식(models/clinical_rna_only.py::ClinicalRNAOnly "
             "참조). concat(기본, 기존 동작) | film(clinical→스칼라 γ,β로 z_rna 전체를 균일 "
             "스케일/이동, 항등 근처 초기화, 파라미터 4개) | cox_add(clinical을 고전적 Cox 가산항 "
             "risk_head(z_rna)+β_age·age_z+β_sex·sex로 직접 더함, 파라미터 2개). findings_backlog.md "
             "15번 항목 — concat이 M6를 못 넘는 게 '신호 없는 clinical을 신호 있는 RNA와 하나의 "
             "선형 결합기에서 공동 학습시키는 구조 자체' 때문인지, 자유도를 극단적으로 줄이면 "
             "해결되는지 검증.",
    )
    parser.add_argument(
        "--clinical-lr-mult", type=float, default=1.0,
        help="2026-09-02: train.py --clinical-lr-mult 이식. combine_mode='cox_add'(--HDP/--HDP-PRETRAIN "
             "포함)에서 clinical_linear(zero-init 선형)가 rna_encoder+risk_head(비선형, 파라미터 훨씬 "
             "많음)와 같은 lr로 경쟁하면 밀린다는 가설 검증용(scripts/diagnose_m7_branch_contrib.py로 "
             "clinical 설명분산비율이 leak 유무와 무관하게 ~0%였던 것 확인, 2026-09-02) — clinical_linear "
             "파라미터그룹의 lr만 base lr의 이 배수로 키운다. 기본 1.0=기존 동작(단일 옵티마이저).",
    )
    parser.add_argument(
        "--rna-lr-mult", type=float, default=1.0,
        help="--clinical-lr-mult와 동일 관례, rna_encoder에 적용(대칭 ablation용).",
    )
    return parser.parse_args()


def main():
    load_env()
    args = _parse_args()
    auc_days = tuple(int(x.strip()) for x in args.auc_days.split(",") if x.strip())
    cfg = Config()
    if args.seed is not None:
        cfg.data.seed  = args.seed
        cfg.light.seed = args.seed
    if args.lr is not None:
        cfg.light.lr = args.lr
    if args.weight_decay is not None:
        cfg.light.weight_decay = args.weight_decay
    if args.epochs is not None:
        cfg.light.epochs = args.epochs
    set_seed(cfg.light.seed)
    device = torch.device(cfg.light.device)
    start_time = datetime.now()

    external_dataset = None
    if args.external:
        if args.dataset == "both":
            raise ValueError("--external은 --dataset both와 함께 쓸 수 없습니다.")
        external_dataset = {"tcga": "cptac", "cptac": "tcga"}[args.dataset]
    if args.eval_external_ckpt and not args.external:
        raise ValueError("--eval-external-ckpt는 --external과 함께 써야 합니다.")

    if args.rna_genes.endswith("_tcga_only") and not (args.dataset == "tcga" and args.external):
        raise ValueError(
            f"--rna-genes {args.rna_genes}는 --dataset tcga --external(TCGA로 학습 -> "
            "CPTAC 전체를 external test)에서만 의미가 있습니다 — TCGA train split만으로 뽑힌 "
            "유전자셋이라 다른 조합에서 쓰면 코호트 불일치로 결과 해석이 잘못됩니다."
        )
    if args.rna_genes.endswith("_cptac_only") and not (args.dataset == "cptac" and args.external):
        # 위 _tcga_only 가드의 반대 방향 버전 — 2026-08-04 추가. CPTAC train split만으로 뽑힌
        # 유전자셋이므로 --dataset cptac --external(CPTAC로 학습 -> TCGA 전체 external test)
        # 조합에서만 leakage-free 전제가 성립한다.
        raise ValueError(
            f"--rna-genes {args.rna_genes}는 --dataset cptac --external(CPTAC로 학습 -> "
            "TCGA 전체를 external test)에서만 의미가 있습니다 — CPTAC train split만으로 뽑힌 "
            "유전자셋이라 다른 조합에서 쓰면 코호트 불일치로 결과 해석이 잘못됩니다."
        )

    if args.raw_linear and not args.M5:
        raise ValueError("--raw-linear는 --M5 전용입니다.")

    with_clinical = args.M5 or args.M7 or args.HDP or args.HDP_PRETRAIN
    with_rna = args.M6 or args.M6X or args.M7 or args.HDP or args.HDP_PRETRAIN

    if with_clinical:
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

    if args.clinical_margin:
        if args.dataset == "both":
            import pandas as pd
            margin_df = pd.concat([
                pd.read_csv(CLINICAL_PATHS["tcga"])[["residual_disease"]],
                pd.read_csv(CLINICAL_PATHS["cptac"])[["residual_disease"]],
            ])
            margin_stats = margin_stats_from_df(margin_df)
        else:
            margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.dataset])
    else:
        margin_stats = None

    if args.clinical_staging:
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

    rna_purist = with_rna and args.rna_genes in ("purist", "purist_top20_tcga_only", "purist_pathway8")
    rna_pathway_categories = None
    if with_rna and args.rna_genes == "pathway8_flat":
        # 2026-09-03: pathway8과 정확히 같은 163개 문헌 큐레이션 유전자를 카테고리 평균(8차원)이
        # 아니라 개별 유전자 z-score 그대로(163차원) 쓴다 — "차원을 줄이는 게 정말 필요한가"를
        # 직접 검증하기 위한 대조군(사용자 지시). rna_pathway_categories=None이라 아래 일반
        # rna_gene_ids 경로(평균 없이 개별 유전자 concat)를 그대로 탄다.
        rna_gene_ids = pathway_flat_gene_ids()
        rna_input_dim = len(rna_gene_ids)
    elif with_rna and args.rna_genes == "purist_pathway8":
        # 2026-09-03: PurIST(1차원, 고정 계수) + pathway8(8차원 카테고리 평균) 하이브리드 —
        # 둘 다 생존 라벨을 전혀 안 봐서(purist_top20_tcga_only와 달리 Cox 선택 유전자가 전혀
        # 안 섞임) 완전히 leak-free인 채로 PDAC RNA 표현력을 pathway8 단독(8차원)보다 키운다
        # (BRCA에서 PAM50+Oncotype DX+pan-cancer 카테고리로 확장한 것과 같은 발상,
        # findings_backlog.md 2026-09-03 참조). data/dataset.py가 rna_purist=True와
        # rna_pathway_categories가 동시에 주어지면 두 feature를 이어붙이도록 처리한다.
        rna_gene_ids = None
        rna_pathway_categories = pathway_category_gene_ids()
        rna_input_dim = 1 + len(rna_pathway_categories)
    elif with_rna and args.rna_genes == "pathway8":
        # 2026-09-02: literature_guided_gene_ids_intersection(Cox test 기반)이 라벨을 보고
        # 유전자를 고르는 방식이라, paper-spec 5-fold CV의 fold-test 환자 중 약 60%가 그
        # 선정 과정의 "train"에 이미 들어가 있었다는 leakage가 확인됨(사용자 지적으로 발견,
        # 정량화는 scripts/diagnose_hdp_checkpoint_weights.py 결과 및 이 세션의 대화 참조).
        # pathway8은 OS 라벨을 전혀 참조하지 않는 순수 문헌 큐레이션(8개 생물학적 카테고리
        # 평균 z-score, train.py --rna-genes pathway8과 동일 — data/dataset.py::
        # pathway_category_gene_ids/WSISurvivalDataset의 rna_pathway_categories 파라미터가
        # 이미 dataset 레벨에서 지원, train.py에서만 배선돼 있었고 이 스크립트(M5/M6/M6X/M7/
        # HDP 공용)엔 없어서 추가함 — leakage 없는 RNA 대안으로 M7부터 비교.
        rna_gene_ids = None
        rna_pathway_categories = pathway_category_gene_ids()
        rna_input_dim = len(rna_pathway_categories)
    elif with_rna and args.rna_genes == "purist_top20_tcga_only":
        # 하이브리드: PurIST 확률(1차원, chance 수준이었음) + TCGA train split Cox-score 상위
        # 20개 개별 유전자(leakage-free) concat. data/dataset.py가 rna_purist=True와
        # rna_gene_ids가 동시에 주어지면 두 feature를 이어붙이도록 처리한다.
        rna_gene_ids = literature_guided_gene_ids_single_cohort("tcga", 20)
        rna_input_dim = 1 + len(rna_gene_ids)
    elif with_rna and rna_purist:
        rna_gene_ids = None
        rna_input_dim = 1
    elif with_rna:
        if args.rna_genes.endswith("_tcga_only") or args.rna_genes.endswith("_cptac_only"):
            # 이 분기를 먼저 안 걸러 아래 일반 분기로 흘려보내면 leaky한 both-결합
            # literature_guided_gene_ids(N)이 조용히 로드된다 — 반드시 여기서 single-cohort/FDR
            # 로더로만 보낸다(data/dataset.py::resolve_tcga_only_rna_genes, train.py와 동일한 관례).
            rna_gene_ids = resolve_tcga_only_rna_genes(args.rna_genes)
        elif args.rna_genes.endswith("_intersection"):
            rna_gene_ids = literature_guided_gene_ids_intersection(int(args.rna_genes.split("_")[1]))
        else:
            rna_gene_ids = (
                pdac_subtype_gene_ids() if args.rna_genes == "subtype"
                else literature_guided_gene_ids(int(args.rna_genes.split("_")[1]))
            )
        rna_input_dim = len(rna_gene_ids)
    else:
        rna_gene_ids, rna_input_dim = None, None

    model_prefix = ("M5" if args.M5 else "M6" if args.M6 else "M6X" if args.M6X else "M7" if args.M7
                     else "HDP_PRETRAIN" if args.HDP_PRETRAIN else "HDP")
    if args.rna_genes == "pathway8_flat":
        # _PW8FLAT = pathway8과 같은 163개 유전자를 평균 없이 개별 z-score로(2026-09-03 추가) —
        # _PW8(8차원 평균)/_EX(leaky both-결합)와 절대 안 섞이게 별도 접미사.
        model_prefix += "_PW8FLAT"
    elif args.rna_genes == "purist_pathway8":
        # _DPW8 = PurIST(1차원) + pathway8(8차원) 하이브리드, 총 9차원 — 순수 _PW8(pathway8만)
        # 및 _D(purist만)와 파일명이 겹치면 안 되므로 별도 접미사(2026-09-03 추가).
        model_prefix += "_DPW8"
    elif args.rna_genes == "pathway8":
        # _PW8 = 문헌 큐레이션 8개 카테고리 평균(leakage 없음) — _INT{n}(Cox test 기반,
        # leakage 있음)과 절대 안 섞이게 별도 접미사.
        model_prefix += "_PW8"
    elif args.rna_genes == "purist_top20_tcga_only":
        # _D20 = PurIST(1차원) + TCGA-only top-20 유전자 하이브리드 — 순수 _D(PurIST만)와
        # 파일명이 겹치면 안 되므로 별도 접미사.
        model_prefix += "_D20"
    elif rna_purist:
        # _D = deterministic RNA pool(PurIST) — gene-selection 없이 고정 계수 분류기 1차원.
        # _EXT{n}/_EX(gene-selection 기반)와 계보가 완전히 다르므로 별도 접미사로 구분한다.
        model_prefix += "_D"
    elif args.rna_genes.endswith("_tcga_only"):
        # train.py와 동일한 관례 — 기존 _EX(leaky, both-결합)와 절대 섞이면 안 된다. N까지
        # 태그에 넣어 EXT500/EXT1500처럼 서로 다른 크기도 섞이지 않게 한다.
        # 2026-09-03: prefix가 "variance"면 literature 기반(_EXT{n})과 파일명이 겹치지 않게
        # _EXTVAR{n}으로 구분한다(같은 n이어도 완전히 다른 유전자 리스트라 절대 섞이면 안 됨).
        _spec = args.rna_genes.split("_")[1]
        _prefix_tag = "VAR" if args.rna_genes.startswith("variance_") else ""
        model_prefix += f"_EXT{_prefix_tag}{_spec}"
    elif args.rna_genes.endswith("_cptac_only"):
        # _tcga_only와 같은 spec(예: fdr0.1)이어도 반대 코호트에서 뽑힌 다른 유전자셋이라
        # CPTAC 접미사로 명시적으로 구분한다 — 안 붙이면 "M7_EXTfdr0.1"이 tcga_only 결과와
        # 파일명이 겹쳐 checkpoint/kfold_preds가 서로 덮어써진다.
        _spec = args.rna_genes.split("_")[1]
        _prefix_tag = "VAR" if args.rna_genes.startswith("variance_") else ""
        model_prefix += f"_EXT{_prefix_tag}{_spec}CPTAC"
    elif args.rna_genes.endswith("_intersection"):
        # _INT{n} = TCGA-only/CPTAC-only 순위 교집합(양방향 leakage-free) 사용 표시.
        model_prefix += f"_INT{args.rna_genes.split('_')[1]}"
    elif args.rna_genes != "subtype":
        model_prefix += "_EX"
    if args.clinical_staging:
        # train.py --clinical-staging과 동일 관례 — _R보다 먼저 붙인다(PMA_..._STG_R_...와 태그 순서 통일).
        model_prefix += "_STG"
    if args.clinical_margin:
        # M5_R — age/sex + 절제연 상태(residual_disease). M5(age/sex만)와 checkpoint/kfold_preds
        # 파일명이 겹치면 안 되므로 별도 접미사.
        model_prefix += "_R"
        if args.no_age_sex:
            # M5_R_ONLY — margin 단독(age/sex 제외).
            model_prefix += "_ONLY"
    if args.clinical_mutation:
        model_prefix += "_MUT"
    if args.raw_linear:
        model_prefix += "_RAWLIN"
    if args.lr is not None and args.lr != 1e-3:
        # _LR{lr} = cfg.light.lr(기본 1e-3) 이외 값 사용 표시 - train.py의 _EX/_SS/_AUX와 같은 관례.
        model_prefix += f"_LR{args.lr:.0e}"
    if args.weight_decay is not None and args.weight_decay != 1e-2:
        model_prefix += f"_WD{args.weight_decay:.0e}"
    if args.match_reference_cohort:
        model_prefix += "_REFCOHORT"
    if args.clinical_dim is not None:
        model_prefix += f"_CLINDIM{args.clinical_dim}"
    if args.rna_dim is not None:
        model_prefix += f"_RNADIM{args.rna_dim}"
    if args.clinical_dropout != 0.0:
        model_prefix += f"_CLINDROP{args.clinical_dropout:g}"
    if args.sex_onehot:
        model_prefix += "_SEXOH"
    if args.combine_mode != "concat":
        model_prefix += f"_{args.combine_mode.upper()}"
    if args.clinical_lr_mult != 1.0:
        model_prefix += f"_CLR{args.clinical_lr_mult:g}"
    if args.rna_lr_mult != 1.0:
        model_prefix += f"_RLR{args.rna_lr_mult:g}"
    if args.risk_hidden_dim is not None:
        model_prefix += f"_RISKDIM{args.risk_hidden_dim}"
    if args.risk_dropout != 0.0:
        model_prefix += f"_RISKDROP{args.risk_dropout:g}"
    if args.fold is not None:
        model_prefix += f"_FOLD{args.fold}OF{args.n_folds}"

    cluster_hist_lookup = None
    if args.HDP or args.HDP_PRETRAIN:
        hist_datasets = ["tcga", "cptac"] if args.dataset == "both" else [args.dataset]
        if external_dataset:
            hist_datasets = list(set(hist_datasets) | {external_dataset})
        hdp_source = "pretrain" if args.HDP_PRETRAIN else "cluster"
        cluster_hist_lookup, hist_k = _load_cluster_histograms(hist_datasets, source=hdp_source)
        print(f"--{model_prefix}: {hdp_source} feature {len(cluster_hist_lookup.lookup)}명 로드 완료 "
              f"(D={hist_k}, {hist_datasets})")

    if args.init_seed is not None:
        torch.manual_seed(args.init_seed)
    if args.M5:
        model = ClinicalOnly(cfg.model, age_mean=age_mean, age_std=age_std,
                              use_margin=args.clinical_margin, margin_stats=margin_stats,
                              use_age_sex=not args.no_age_sex,
                              use_staging=args.clinical_staging, stage_stats=stage_stats,
                              raw_linear=args.raw_linear).to(device)
    elif args.M6:
        model = RNAOnly(cfg.model, rna_input_dim=rna_input_dim).to(device)
    elif args.M6X:
        model = RNAOnlyExtend(cfg.model, rna_input_dim=rna_input_dim).to(device)
    elif args.HDP or args.HDP_PRETRAIN:
        model = HDP(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                    hist_dim=hist_k, use_margin=args.clinical_margin, margin_stats=margin_stats,
                    use_age_sex=not args.no_age_sex,
                    use_staging=args.clinical_staging, stage_stats=stage_stats).to(device)
    else:
        model = ClinicalRNAOnly(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                                 clinical_dim=args.clinical_dim, rna_dim=args.rna_dim,
                                 clinical_dropout=args.clinical_dropout, sex_onehot=args.sex_onehot,
                                 combine_mode=args.combine_mode, risk_hidden_dim=args.risk_hidden_dim,
                                 risk_dropout=args.risk_dropout, use_margin=args.clinical_margin,
                                 margin_stats=margin_stats, use_age_sex=not args.no_age_sex,
                                 use_staging=args.clinical_staging, stage_stats=stage_stats,
                                 use_mutation=args.clinical_mutation, mutation_stats=mutation_stats).to(device)
    if args.init_seed is not None:
        torch.manual_seed(cfg.light.seed)

    run_ts = datetime.now().strftime("%m%d::%H%M")
    group_ts = args.group_ts or run_ts
    wandb_group = f"{model_prefix}_{group_ts}"
    if WANDB_AVAILABLE and not args.eval_external_ckpt:
        run_name = f"{args.dataset.upper()}_{model_prefix}_seed{cfg.light.seed}_{run_ts}"
        wandb.init(
            project="Path-ViT",
            name=run_name,
            group=wandb_group,
            config={
                "epochs": cfg.light.epochs, "lr": cfg.light.lr, "weight_decay": cfg.light.weight_decay,
                "seed": cfg.light.seed, "warmup_epochs": cfg.light.warmup_epochs,
                "cox_batch_size": cfg.light.cox_batch_size,
                "embed_dim": cfg.model.embed_dim, "dropout": cfg.model.dropout,
                "model": model_prefix, "rna_input_dim": rna_input_dim, "rna_genes": args.rna_genes,
                "dataset": args.dataset, "external_dataset": external_dataset,
            },
        )

    restrict_case_ids = None
    if args.match_reference_cohort:
        from data.reference_cohort import reference_eligible_case_ids
        target_datasets = ["tcga", "cptac"] if args.dataset == "both" else [args.dataset]
        if external_dataset:
            target_datasets = list(set(target_datasets) | {external_dataset})
        restrict_case_ids = reference_eligible_case_ids(target_datasets, cfg=cfg.data)
        print(f"--match-reference-cohort: {len(restrict_case_ids)}개 case로 제한")

    ds_kwargs = dict(with_clinical=with_clinical, with_margin=args.clinical_margin, with_staging=args.clinical_staging,
                      with_mutation=args.clinical_mutation,
                      with_rna=with_rna, rna_gene_ids=rna_gene_ids, rna_purist=rna_purist,
                      rna_pathway_categories=rna_pathway_categories,
                      restrict_case_ids=restrict_case_ids)
    split_kwargs = dict(fold=args.fold, n_folds=args.n_folds)
    train_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split=("all" if args.full_train else "train"),
                                   **ds_kwargs, **split_kwargs)
    val_ds   = None if args.full_train else WSISurvivalDataset(cfg.data, dataset=args.dataset, split="val",   **ds_kwargs, **split_kwargs)
    test_ds  = None if args.full_train else WSISurvivalDataset(cfg.data, dataset=args.dataset, split="test",  **ds_kwargs, **split_kwargs)
    external_ds = (
        WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", **ds_kwargs)
        if external_dataset else None
    )

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    train_loader      = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **dl_kwargs)
    val_loader        = DataLoader(val_ds,   shuffle=False, **dl_kwargs) if val_ds  is not None else None
    test_loader       = DataLoader(test_ds,  shuffle=False, **dl_kwargs) if test_ds is not None else None
    external_loader   = DataLoader(external_ds, shuffle=False, **dl_kwargs) if external_ds else None

    if args.eval_external_ckpt:
        # [--eval-external-ckpt] train.py와 동일 관례 — 학습을 건너뛰고 이미 저장된 checkpoint의
        # external 예측만 다시 뽑는다. model/dataset은 위에서 이미 정상 경로로 생성됐으니(구조
        # 불일치 위험 없음) 여기서 state_dict만 얹고 evaluate()를 한 번 호출한 뒤 즉시 종료한다.
        if external_loader is None:
            raise ValueError("--eval-external-ckpt는 --external과 함께 써야 합니다.")
        ckpt = torch.load(args.eval_external_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"--eval-external-ckpt: {args.eval_external_ckpt} 로드 완료 "
              f"(epoch={ckpt.get('epoch')}, val_c_index={ckpt.get('val_c_index')})")
        external_metrics = evaluate(model, external_loader, device, cluster_hist_lookup)
        print(f"  external c_index={external_metrics['c_index']:.4f} | HR={external_metrics['hr']:.3f} "
              f"[{external_metrics['hr_ci_lower']:.3f}, {external_metrics['hr_ci_upper']:.3f}] | "
              f"log_rank_p={external_metrics['log_rank_p']:.4f}")
        import csv
        pred_dir = Path(__file__).parent / ".logs" / "external_preds"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / f"{external_dataset}_{model_prefix}_seed{cfg.light.seed}_fold{args.fold}of{args.n_folds}.csv"
        with open(pred_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
            for cid, risk, t, e in zip(external_metrics["case_ids"], external_metrics["risks"],
                                        external_metrics["times"], external_metrics["events"]):
                writer.writerow([cid, risk, t, e])
        print(f"  -> external predictions saved: {pred_path}")
        return

    split_desc = f"{args.n_folds}-fold CV, fold {args.fold}" if args.fold is not None else "6:2:2 stratified split"
    print(f"Model: {model_prefix} ({type(model).__name__}) | params={sum(p.numel() for p in model.parameters()):,}")
    if args.full_train:
        print(f"Dataset: {args.dataset}  (--full-train: split 없이 코호트 전체를 train으로 사용)  "
              f"Train: {len(train_ds)} patients (val/internal test 없음)")
    else:
        print(f"Dataset: {args.dataset}  ({split_desc})  "
              f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test(internal): {len(test_ds)} patients")
    print(f"lr={cfg.light.lr:.1e} | weight_decay={cfg.light.weight_decay:.1e} | "
          f"epochs={cfg.light.epochs} | cox_batch_size={cfg.light.cox_batch_size}")

    if args.clinical_lr_mult != 1.0 or args.rna_lr_mult != 1.0:
        groups = _branch_param_groups(model)
        param_groups = []
        if groups["clinical"]:
            param_groups.append({"params": groups["clinical"], "lr": cfg.light.lr * args.clinical_lr_mult})
        if groups["rna"]:
            param_groups.append({"params": groups["rna"], "lr": cfg.light.lr * args.rna_lr_mult})
        if groups["other"]:
            param_groups.append({"params": groups["other"], "lr": cfg.light.lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.light.weight_decay)
        print(f"branch-lr-mult 적용: clinical={args.clinical_lr_mult}x({len(groups['clinical'])}개 텐서), "
              f"rna={args.rna_lr_mult}x({len(groups['rna'])}개 텐서), other=1x({len(groups['other'])}개 텐서)")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.light.lr, weight_decay=cfg.light.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg)

    ckpt_dir = Path(__file__).parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # 2026-08-03: train.py는 checkpoint 파일명에 seed를 넣는데(tag += f"_seed{...}") 이 스크립트는
    # 안 넣고 있었다 — 같은 model_prefix로 시드만 바꿔 돌리면 서로 덮어써 앙상블(scripts/
    # ensemble_eval.py류)에 쓸 시드별 체크포인트가 남지 않는 문제. train.py 관례에 맞춘다.
    ckpt_path = ckpt_dir / f"survival_{args.dataset}_best_{model_prefix.lower()}_seed{cfg.light.seed}_light.pt"

    best_score, best_metrics = -1.0, {}
    epochs_since_improvement = 0
    for epoch in range(cfg.light.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(model, train_loader, optimizer, device, cfg.light.cox_batch_size, cluster_hist_lookup)
        train_metrics = evaluate(model, train_eval_loader, device, cluster_hist_lookup)
        scheduler.step()

        if val_ds is None:
            # --full-train: val 자체가 없음 — train 지표만 기록하고 다음 epoch으로.
            print(f"Epoch {epoch+1:3d} | lr={lr_now:.2e} | loss={loss:.4f} | "
                  f"train_c_index={train_metrics['c_index']:.4f}")
            if WANDB_AVAILABLE:
                wandb.log({
                    "train/loss": loss, "train/lr": lr_now, "train/c_index": train_metrics["c_index"],
                }, step=epoch + 1)
            continue

        metrics = evaluate(model, val_loader, device, cluster_hist_lookup)
        val_td_auc = compute_time_dependent_auc(
            train_metrics["times"], train_metrics["events"], metrics["times"], metrics["events"], metrics["risks"],
            eval_days=auc_days,
        )

        c_index = metrics.get("c_index", float("nan"))
        score = c_index if not math.isnan(c_index) else -1.0
        print(f"Epoch {epoch+1:3d} | lr={lr_now:.2e} | loss={loss:.4f} | "
              f"train_c_index={train_metrics['c_index']:.4f} | " + _log_line("val", metrics, val_td_auc))

        if WANDB_AVAILABLE:
            wandb.log({
                "train/loss": loss, "train/lr": lr_now, "train/c_index": train_metrics["c_index"],
                "val_performance/c_index": metrics["c_index"], "val_performance/hr": metrics["hr"],
                "val_performance/log_rank_p": metrics["log_rank_p"],
                "val_performance/auc_mean": val_td_auc["auc_mean"],
            }, step=epoch + 1)

        if score > best_score:
            best_score = score
            best_metrics = {**metrics, "epoch": epoch + 1}
            epochs_since_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_c_index": best_score}, ckpt_path)
            print(f"  -> checkpoint saved (c_index={best_score:.4f})")
            if WANDB_AVAILABLE:
                wandb.run.summary["best_val_c_index"] = best_score
                wandb.run.summary["best_epoch"] = epoch + 1
        else:
            epochs_since_improvement += 1
            if args.patience is not None and epochs_since_improvement >= args.patience:
                print(f"  -> early stopping (patience={args.patience}, "
                      f"best epoch {best_metrics.get('epoch', '-')} c_index={best_score:.4f})")
                break

    if not args.full_train:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        train_metrics_final = evaluate(model, train_eval_loader, device, cluster_hist_lookup)
        test_metrics = evaluate(model, test_loader, device, cluster_hist_lookup)
        test_td_auc = compute_time_dependent_auc(
            train_metrics_final["times"], train_metrics_final["events"],
            test_metrics["times"], test_metrics["events"], test_metrics["risks"],
            eval_days=auc_days,
        )
        print(f"\n=== Internal Test (best checkpoint epoch {ckpt['epoch']}) ===")
        print(_log_line("test", test_metrics, test_td_auc))

        if args.fold is not None:
            import csv
            pred_dir = Path(__file__).parent / ".logs" / "kfold_preds"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"{args.dataset}_{model_prefix}_seed{cfg.light.seed}_fold{args.fold}of{args.n_folds}.csv"
            with open(pred_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
                for cid, risk, t, e in zip(test_metrics["case_ids"], test_metrics["risks"],
                                            test_metrics["times"], test_metrics["events"]):
                    writer.writerow([cid, risk, t, e])
            print(f"  -> fold predictions saved: {pred_path}")

        if WANDB_AVAILABLE:
            wandb.run.summary["test_c_index"] = test_metrics["c_index"]
            wandb.run.summary["test_hr"] = test_metrics["hr"]
            wandb.run.summary["test_log_rank_p"] = test_metrics["log_rank_p"]
            wandb.run.summary["test_auc_mean"] = test_td_auc["auc_mean"]
            wandb.finish()
    else:
        # --full-train: val/test 자체가 없어 best-val 체크포인트 선택이 불가능하므로, 학습 루프
        # 종료 직후(메모리 상의 model이 곧 마지막 epoch 가중치) train_metrics를 그대로 재사용하고
        # model도 리로드하지 않는다 — 아래 external 평가가 자연히 "마지막 epoch 모델" 기준으로
        # 정확히 1회 수행된다(train.py --full-train와 동일 관례).
        train_metrics_final = train_metrics
        if WANDB_AVAILABLE:
            wandb.finish()

    external_metrics, external_td_auc = None, None
    if external_ds is not None:
        external_metrics = evaluate(model, external_loader, device, cluster_hist_lookup)
        external_td_auc = compute_time_dependent_auc(
            train_metrics_final["times"], train_metrics_final["events"],
            external_metrics["times"], external_metrics["events"], external_metrics["risks"],
            eval_days=auc_days,
        )
        print(f"\n=== External Test ({external_dataset} 전체 코호트) ===")
        print(_log_line("external", external_metrics, external_td_auc))
        if args.full_train:
            # 2026-09-02: --full-train은 checkpoint 저장 자체가 없어(위 969행 주석 — val이 없어
            # "best" 선택이 불가능) --eval-external-ckpt로 나중에 CSV를 다시 뽑을 방법이 없다.
            # 이 실행 안에서 바로 저장한다 — fold 개념이 없으므로 파일명에 fold 표기 없이
            # seed만으로 구분(여러 seed로 반복해 external 평균±CI를 보는 용도,
            # scripts/pool_fulltrain_external_preds.py 입력용).
            import csv
            pred_dir = Path(__file__).parent / ".logs" / "external_preds"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"{external_dataset}_{model_prefix}_FULLTRAIN_seed{cfg.light.seed}.csv"
            with open(pred_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
                for cid, risk, t, e in zip(external_metrics["case_ids"], external_metrics["risks"],
                                            external_metrics["times"], external_metrics["events"]):
                    writer.writerow([cid, risk, t, e])
            print(f"  -> external predictions saved: {pred_path}")
        if WANDB_AVAILABLE:
            wandb.init(
                project="Path-ViT",
                name=f"{args.dataset.upper()}_X{model_prefix}_seed{cfg.light.seed}_{run_ts}",
                group=wandb_group,
                config={"dataset": args.dataset, "external_dataset": external_dataset, "model": model_prefix},
            )
            wandb.log({
                "external/c_index": external_metrics["c_index"], "external/hr": external_metrics["hr"],
                "external/log_rank_p": external_metrics["log_rank_p"],
                "external/auc_mean": external_td_auc["auc_mean"],
            })
            wandb.finish()

    elapsed = datetime.now() - start_time
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    external_line = (
        f"> External({external_dataset.upper()}) C-index: *{external_metrics['c_index']:.4f}*\n"
        if external_metrics is not None else ""
    )
    internal_line = (
        "> Internal: --full-train(split 없음, internal test 없음)\n" if args.full_train
        else f"> Internal Test C-index: *{test_metrics['c_index']:.4f}*\n"
    )
    send_slack(
        f":white_check_mark: *Path-ViT-light ({args.dataset.upper()} OS, {model_prefix}) 학습 완료*\n"
        f"{internal_line}"
        f"{external_line}"
        f"> 소요 시간: {h}h {m}m {s}s"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        load_env()
        send_slack(f":x: *Path-ViT-light 학습 에러*\n```{type(e).__name__}: {e}```")
        raise
