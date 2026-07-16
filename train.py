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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, pdac_subtype_gene_ids
from models import ViT_M1, LateFusionViT, ViT_M2, ViT_M4
from models.clinical_encoder import age_stats_from_csv
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


def _patient_risk(model, patient_slides, device, amp_ctx, transform, chunk_size) -> torch.Tensor:
    """환자 1명이 보유한 슬라이드 전부를 forward해 임베딩을 평균 풀링한 뒤 risk score(scalar)를 계산한다.

    [--M2/--M4] model이 clinical_encoder(및 rna_encoder)를 보유하면, age/sex(/rna)는
    슬라이드가 아니라 환자 단위 메타데이터이므로 슬라이드 평균 풀링 이후
    combine_with_clinical()(--M2) 또는 combine_with_clinical_rna()(--M4)로 결합한다.
    """
    with amp_ctx:
        slide_embeds = []
        for slide in patient_slides:
            coords = slide["coords"].to(device, non_blocking=True)
            if "features" in slide:
                out = model(coords, features=slide["features"])
            else:
                out = model(coords, patch_paths=slide["patch_paths"],
                             transform=transform, chunk_size=chunk_size)
            slide_embeds.append(out["embed"])

        patient_embed = torch.stack(slide_embeds).mean(dim=0)      # (D,) 또는 (2D,)/(3D,) — 슬라이드 평균 풀링

        if hasattr(model, "rna_encoder"):
            age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
            sex_idx   = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
            rna       = patient_slides[0]["rna"].to(device, non_blocking=True)
            patient_embed = model.combine_with_clinical_rna(patient_embed, age_years, sex_idx, rna)  # (3D,)
        elif hasattr(model, "clinical_encoder"):
            age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
            sex_idx   = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
            patient_embed = model.combine_with_clinical(patient_embed, age_years, sex_idx)  # (2D,)

        risk = model.risk_head(patient_embed.unsqueeze(0)).view(1)  # (1,)
    return risk


def train_one_epoch(
    model, loader, optimizer, cfg, device, amp_ctx, transform
) -> float:
    model.train()
    if model.cnn.backbone is not None:
        model.cnn.backbone.eval()  # frozen backbone의 BN을 population stats(eval)로 고정 — train/eval 분포 불일치 방지
    total_loss    = 0.0
    total_batches = 0
    chunk_size    = cfg.train.cnn_chunk_size
    batch_size    = cfg.train.cox_batch_size

    risks, times, events = [], [], []

    def _flush():
        nonlocal risks, times, events, total_loss, total_batches
        if not risks:
            return
        risk_t  = torch.cat(risks)
        time_t  = torch.cat(times).to(device)
        event_t = torch.cat(events).to(device)

        loss = cox_ph_loss(risk_t, time_t, event_t)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item()
        total_batches += 1
        risks, times, events = [], [], []

    for patient_slides in loader:                # 환자 1명 분량의 슬라이드 리스트
        if len(patient_slides) == 0:
            continue
        risk = _patient_risk(model, patient_slides, device, amp_ctx, transform, chunk_size)

        risks.append(risk)
        times.append(patient_slides[0]["OS_time"])
        events.append(patient_slides[0]["OS_event"])

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
        risk = _patient_risk(model, patient_slides, device, amp_ctx, transform, chunk_size)

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
    # [LateFusion] --fusion 플래그로 LateFusionViT 사용 여부 선택
    # 미지정 시 기존 ViT_M1(ViT+ABMIL)로 동작 — ablation baseline 유지
    parser.add_argument(
        "--fusion", action="store_true",
        help="LateFusionViT 사용 (ViT+ABMIL + Cluster Histogram). "
             "data/fit_clusters.py 실행으로 cluster_centroids.pt 사전 생성 필요.",
    )
    # [Clinical/RNA] --M1/--M2/--M4로 모델 종류 선택 (상호 배타)
    # --M1(기본값): 순수 WSI 모델(ViT_M1, --fusion 지정 시 LateFusionViT)
    # --M2        : ViT_M2 — WSI 임베딩 + Clinical(age/sex) MLP Late Fusion 멀티모달
    # --M4        : ViT_M4 — WSI + Clinical(age/sex) + RNA-seq MLP 3-모달 Late Fusion
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
             "3-모달 Late Fusion). data/clinical_{tcga,cptac}.csv, data/rna_{tcga,cptac}.csv "
             "필요. --fusion과 동시 사용 불가.",
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

    # [LateFusion] --fusion 플래그 시 cluster_centroids.pt 로드 검증
    if args.fusion and not cfg.data.precomputed:
        raise ValueError("--fusion은 precomputed(features.pt) 모드에서만 지원됩니다. --image와 함께 사용 불가.")
    if args.M2 and args.fusion:
        raise ValueError("--M2(Clinical fusion)와 --fusion(Cluster fusion)은 동시에 지원되지 않습니다.")
    if args.M4 and args.fusion:
        raise ValueError("--M4(Clinical+RNA fusion)와 --fusion(Cluster fusion)은 동시에 지원되지 않습니다.")
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

    # [Clinical] --M2/--M4 시 age z-score 정규화 통계를 학습 코호트(args.dataset)에서 계산해
    # 고정한다(extract_rna_clinical.py의 "데이터셋 내부 z-score 정규화" 관례와 동일).
    # dataset="both"면 두 코호트 clinical.csv를 합쳐 통계를 계산한다.
    if args.M2 or args.M4:
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

    # [RNA] --M4 시 RNAEncoder 입력 차원 = Bailey 2016 + Moffitt 2015 PDAC subtype 분류
    # 유전자 수(data/dataset.py::pdac_subtype_gene_ids(), WSISurvivalDataset(with_rna=True)가
    # 실제 로드하는 유전자 컬럼과 동일한 기준).
    rna_input_dim = len(pdac_subtype_gene_ids()) if args.M4 else None

    set_seed(cfg.train.seed)
    start_time = datetime.now()
    device = torch.device(cfg.train.device)

    torch.backends.cudnn.benchmark = True

    amp_ctx = _make_amp_ctx()

    if args.M4:
        model_prefix = "M4"
    elif args.M2:
        model_prefix = "M2"
    elif args.fusion:
        model_prefix = "M1C"
    else:
        model_prefix = "M1"

    # internal(main) run과 external run이 같은 학습 세션임을 알아볼 수 있도록 timestamp를 공유한다.
    run_ts = datetime.now().strftime("%m%d::%H%M")
    if WANDB_AVAILABLE:
        run_name = f"{args.dataset.upper()}_{model_prefix}_seed{cfg.train.seed}_{run_ts}"
        wandb.init(
            project="Path-ViT",
            name=run_name,
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
                                           else "ViT_M2" if args.M2
                                           else "LateFusionViT" if args.fusion else "ViT_M1"),
                "num_clusters":          int(cluster_centroids.shape[0]) if args.fusion else 0,
                "age_mean":              age_mean,
                "age_std":               age_std,
                "rna_input_dim":         rna_input_dim,
                "dataset":               args.dataset,
                "external_dataset":      external_dataset,
            },
        )

    with_clinical = args.M2 or args.M4
    train_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="train", with_clinical=with_clinical, with_rna=args.M4)
    val_ds   = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="val",   with_clinical=with_clinical, with_rna=args.M4)
    test_ds  = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="test",  with_clinical=with_clinical, with_rna=args.M4)
    # [ExternalTest] 학습에 전혀 쓰이지 않은 코호트 전체(split="all") — 없으면 None
    external_ds = (
        WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", with_clinical=with_clinical, with_rna=args.M4)
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

    # [Clinical/RNA/LateFusion] --M1/--M2/--M4/--fusion에 따라 모델 선택
    # ViT_M1        : 순수 WSI ViT+ABMIL 단일 경로 (--M1, ablation baseline)
    # LateFusionViT : ViT+ABMIL (Path A) + Cluster Histogram (Path B) Late Fusion (--M1 --fusion)
    # ViT_M2        : ViT+ABMIL (WSI) + Clinical age/sex MLP Late Fusion 멀티모달 (--M2)
    # ViT_M4        : ViT+ABMIL (WSI) + Clinical age/sex MLP + RNA-seq MLP 3-모달 Late Fusion (--M4)
    if args.M4:
        model = ViT_M4(cfg.model, age_mean=age_mean, age_std=age_std,
                        rna_input_dim=rna_input_dim, precomputed=cfg.data.precomputed).to(device)
    elif args.M2:
        model = ViT_M2(cfg.model, age_mean=age_mean, age_std=age_std,
                        precomputed=cfg.data.precomputed).to(device)
    elif args.fusion:
        model = LateFusionViT(cfg.model, cluster_centroids, precomputed=cfg.data.precomputed).to(device)
    else:
        model = ViT_M1(cfg.model, precomputed=cfg.data.precomputed).to(device)
    if model.cnn.backbone is not None:
        model.cnn.backbone.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg)

    mode = "precomputed features" if cfg.data.precomputed else "raw image (--image)"
    print(f"Mode: {mode}")
    # [Clinical/RNA/LateFusion] 모델 종류 출력
    if args.M4:
        print(f"Model: ViT_M4 (ViT+ABMIL + Clinical age/sex MLP + RNA-seq MLP, "
              f"age_mean={age_mean:.1f}, age_std={age_std:.1f}, rna_input_dim={rna_input_dim})")
    elif args.M2:
        print(f"Model: ViT_M2 (ViT+ABMIL + Clinical age/sex MLP, "
              f"age_mean={age_mean:.1f}, age_std={age_std:.1f})")
    elif args.fusion:
        K = int(cluster_centroids.shape[0])
        print(f"Model: LateFusionViT (ViT+ABMIL + ClusterHistogram, K={K})")
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
    # [Clinical/RNA/LateFusion] 모델 종류별로 별도 checkpoint 저장 — ablation 결과 보존
    tag = args.dataset
    if args.M4:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_clinical_rna.pt"
    elif args.M2:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_clinical.pt"
    elif args.fusion:
        ckpt_path = ckpt_dir / f"survival_{tag}_best_fusion.pt"
    else:
        ckpt_path = ckpt_dir / f"survival_{tag}_best.pt"

    best_score   = -1.0
    best_metrics = {}
    for epoch in range(cfg.train.epochs):
        lr_now        = optimizer.param_groups[0]["lr"]
        loss          = train_one_epoch(model, train_loader, optimizer, cfg, device, amp_ctx, train_ds.transform)
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
                config={
                    "dataset":          args.dataset,
                    "external_dataset": external_dataset,
                    "model":            ("ViT_M4" if args.M4
                                          else "ViT_M2" if args.M2
                                          else "LateFusionViT" if args.fusion else "ViT_M1"),
                },
            )
            wandb.run.summary["external_dataset"]     = external_dataset
            wandb.run.summary["external_c_index"]     = external_metrics["c_index"]
            wandb.run.summary["external_hr"]          = external_metrics["hr"]
            wandb.run.summary["external_hr_ci_lower"] = external_metrics["hr_ci_lower"]
            wandb.run.summary["external_hr_ci_upper"] = external_metrics["hr_ci_upper"]
            wandb.run.summary["external_log_rank_p"]  = external_metrics["log_rank_p"]
            wandb.run.summary["external_auc_mean"]    = external_td_auc["auc_mean"]
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
