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
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, pdac_subtype_gene_ids, literature_guided_gene_ids
from models import ClinicalOnly, RNAOnly, RNAOnlyExtend, ClinicalRNAOnly
from models.clinical_encoder import age_stats_from_csv
from utils import load_env, send_slack
from utils.losses import cox_ph_loss
from utils.metrics import compute_survival_metrics, compute_time_dependent_auc


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _patient_risk(model, patient_slides, device) -> torch.Tensor:
    """WSI 없이 환자 단위 메타데이터(age/sex 및/또는 rna)만으로 risk score를 계산한다.

    model이 ClinicalRNAOnly(M7)면 age/sex/rna 전부, rna_encoder만 있으면
    RNAOnly/RNAOnlyExtend(M6/M6X), 그 외(clinical_encoder만)는 ClinicalOnly(M5) —
    forward 시그니처가 모델마다 다르므로 분기한다. 2026-07-29: ClinicalRNAOnly의
    combine_mode="film"/"cox_add"는 clinical_encoder 속성이 없어(그 대신 film_gamma/
    clinical_linear) hasattr(model, "clinical_encoder")만으로는 M7을 못 잡는 버그가 있었다 —
    isinstance로 명시적으로 잡는다.
    """
    p = patient_slides[0]
    is_m7 = isinstance(model, ClinicalRNAOnly)
    has_rna = hasattr(model, "rna_encoder")
    if is_m7:
        return model(
            p["age_years"].to(device, non_blocking=True),
            p["sex_idx"].to(device, non_blocking=True),
            p["rna"].to(device, non_blocking=True),
        )
    if has_rna:
        return model(p["rna"].to(device, non_blocking=True))
    return model(
        p["age_years"].to(device, non_blocking=True),
        p["sex_idx"].to(device, non_blocking=True),
    )


def train_one_epoch(model, loader, optimizer, device, batch_size: int) -> float:
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
        risks.append(_patient_risk(model, patient_slides, device))
        times.append(patient_slides[0]["OS_time"])
        events.append(patient_slides[0]["OS_event"])
        if len(risks) >= batch_size:
            _flush()
    _flush()
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_risks, all_times, all_events = [], [], []
    for patient_slides in loader:
        if len(patient_slides) == 0:
            continue
        risk = _patient_risk(model, patient_slides, device)
        all_risks.append(risk.float().item())
        all_times.append(float(patient_slides[0]["OS_time"].item()))
        all_events.append(int(patient_slides[0]["OS_event"].item()))
    risks, times, events = np.array(all_risks), np.array(all_times), np.array(all_events)
    return {**compute_survival_metrics(risks, times, events), "risks": risks, "times": times, "events": events}


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
        "--rna-genes", type=str, default="subtype",
        choices=["subtype", "literature_1000", "literature_1500", "literature_2000"],
        help="RNA 브랜치(--M6/--M6X/--M7) 입력 유전자셋 선택. train.py --rna-genes와 동일한 관례 "
             "(subtype 외 선택 시 wandb/checkpoint에 _EX 접미사 자동 부착).",
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
    parser.add_argument(
        "--clinical-dim", type=int, default=None,
        help="--M7 전용: clinical_encoder 출력 차원(기본 None=모듈 상수 16, models/"
             "clinical_rna_only.py). RNA(256)와 같은 폭(예: 64)으로 키워서 '정보(표현력)를 "
             "더 주면 오히려 나빠지는지' 검증하는 ablation용 — M5(RNA만)가 M7(RNA+Clinical)을 "
             "이긴 관찰(2026-07-28) 이후 clinical 표현력 증가가 해로운지 확인하기 위한 용도.",
    )
    parser.add_argument(
        "--rna-dim", type=int, default=None,
        help="--M7 전용: rna_encoder 출력 차원(기본 None=모듈 상수 256). --clinical-dim과 같이 "
             "써서(예: --rna-dim 64 --clinical-dim 4) 레퍼런스의 RNA:Clinical=16:1 비율은 "
             "유지한 채, 이 프로젝트의 다른 모델(M1~M6, 전부 64차원)과 같은 절대 폭으로 맞춰볼 "
             "때 사용한다.",
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
        "--combine-mode", type=str, default="concat", choices=["concat", "film", "cox_add"],
        help="--M7 전용: clinical과 RNA를 합치는 방식(models/clinical_rna_only.py::ClinicalRNAOnly "
             "참조). concat(기본, 기존 동작) | film(clinical→스칼라 γ,β로 z_rna 전체를 균일 "
             "스케일/이동, 항등 근처 초기화, 파라미터 4개) | cox_add(clinical을 고전적 Cox 가산항 "
             "risk_head(z_rna)+β_age·age_z+β_sex·sex로 직접 더함, 파라미터 2개). findings_backlog.md "
             "15번 항목 — concat이 M6를 못 넘는 게 '신호 없는 clinical을 신호 있는 RNA와 하나의 "
             "선형 결합기에서 공동 학습시키는 구조 자체' 때문인지, 자유도를 극단적으로 줄이면 "
             "해결되는지 검증.",
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

    with_clinical = args.M5 or args.M7
    with_rna = args.M6 or args.M6X or args.M7

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

    if with_rna:
        rna_gene_ids = (
            pdac_subtype_gene_ids() if args.rna_genes == "subtype"
            else literature_guided_gene_ids(int(args.rna_genes.split("_")[1]))
        )
        rna_input_dim = len(rna_gene_ids)
    else:
        rna_gene_ids, rna_input_dim = None, None

    model_prefix = "M5" if args.M5 else "M6" if args.M6 else "M6X" if args.M6X else "M7"
    if args.rna_genes != "subtype":
        model_prefix += "_EX"
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
    if args.risk_hidden_dim is not None:
        model_prefix += f"_RISKDIM{args.risk_hidden_dim}"
    if args.risk_dropout != 0.0:
        model_prefix += f"_RISKDROP{args.risk_dropout:g}"

    if args.init_seed is not None:
        torch.manual_seed(args.init_seed)
    if args.M5:
        model = ClinicalOnly(cfg.model, age_mean=age_mean, age_std=age_std).to(device)
    elif args.M6:
        model = RNAOnly(cfg.model, rna_input_dim=rna_input_dim).to(device)
    elif args.M6X:
        model = RNAOnlyExtend(cfg.model, rna_input_dim=rna_input_dim).to(device)
    else:
        model = ClinicalRNAOnly(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                                 clinical_dim=args.clinical_dim, rna_dim=args.rna_dim,
                                 clinical_dropout=args.clinical_dropout, sex_onehot=args.sex_onehot,
                                 combine_mode=args.combine_mode, risk_hidden_dim=args.risk_hidden_dim,
                                 risk_dropout=args.risk_dropout).to(device)
    if args.init_seed is not None:
        torch.manual_seed(cfg.light.seed)

    run_ts = datetime.now().strftime("%m%d::%H%M")
    group_ts = args.group_ts or run_ts
    wandb_group = f"{model_prefix}_{group_ts}"
    if WANDB_AVAILABLE:
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

    ds_kwargs = dict(with_clinical=with_clinical, with_rna=with_rna, rna_gene_ids=rna_gene_ids,
                      restrict_case_ids=restrict_case_ids)
    train_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="train", **ds_kwargs)
    val_ds   = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="val",   **ds_kwargs)
    test_ds  = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="test",  **ds_kwargs)
    external_ds = (
        WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", **ds_kwargs)
        if external_dataset else None
    )

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    train_loader      = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **dl_kwargs)
    val_loader        = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader       = DataLoader(test_ds,  shuffle=False, **dl_kwargs)
    external_loader   = DataLoader(external_ds, shuffle=False, **dl_kwargs) if external_ds else None

    print(f"Model: {model_prefix} ({type(model).__name__}) | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset: {args.dataset}  (6:2:2 stratified split)  "
          f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test(internal): {len(test_ds)} patients")
    print(f"lr={cfg.light.lr:.1e} | weight_decay={cfg.light.weight_decay:.1e} | "
          f"epochs={cfg.light.epochs} | cox_batch_size={cfg.light.cox_batch_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.light.lr, weight_decay=cfg.light.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg)

    ckpt_dir = Path(__file__).parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"survival_{args.dataset}_best_{model_prefix.lower()}_light.pt"

    best_score, best_metrics = -1.0, {}
    epochs_since_improvement = 0
    for epoch in range(cfg.light.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(model, train_loader, optimizer, device, cfg.light.cox_batch_size)
        train_metrics = evaluate(model, train_eval_loader, device)
        metrics = evaluate(model, val_loader, device)
        val_td_auc = compute_time_dependent_auc(
            train_metrics["times"], train_metrics["events"], metrics["times"], metrics["events"], metrics["risks"],
            eval_days=auc_days,
        )
        scheduler.step()

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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    train_metrics_final = evaluate(model, train_eval_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    test_td_auc = compute_time_dependent_auc(
        train_metrics_final["times"], train_metrics_final["events"],
        test_metrics["times"], test_metrics["events"], test_metrics["risks"],
        eval_days=auc_days,
    )
    print(f"\n=== Internal Test (best checkpoint epoch {ckpt['epoch']}) ===")
    print(_log_line("test", test_metrics, test_td_auc))
    if WANDB_AVAILABLE:
        wandb.run.summary["test_c_index"] = test_metrics["c_index"]
        wandb.run.summary["test_hr"] = test_metrics["hr"]
        wandb.run.summary["test_log_rank_p"] = test_metrics["log_rank_p"]
        wandb.run.summary["test_auc_mean"] = test_td_auc["auc_mean"]
        wandb.finish()

    external_metrics, external_td_auc = None, None
    if external_ds is not None:
        external_metrics = evaluate(model, external_loader, device)
        external_td_auc = compute_time_dependent_auc(
            train_metrics_final["times"], train_metrics_final["events"],
            external_metrics["times"], external_metrics["events"], external_metrics["risks"],
            eval_days=auc_days,
        )
        print(f"\n=== External Test ({external_dataset} 전체 코호트) ===")
        print(_log_line("external", external_metrics, external_td_auc))
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
    send_slack(
        f":white_check_mark: *Path-ViT-light ({args.dataset.upper()} OS, {model_prefix}) 학습 완료*\n"
        f"> Internal Test C-index: *{test_metrics['c_index']:.4f}*\n"
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
