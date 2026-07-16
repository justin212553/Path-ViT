"""
Clinical+RNA 전용 baseline — WSI(병리) 없이 age/sex + RNA-seq만으로 OS risk를 예측한다.

목적: 지금까지 ViT_M1(WSI branch)에서 관찰된 seed별 c-index 변동(0.46~0.62)이
ViT 아키텍처의 복잡도(파라미터 ~20만 개, 환자 90~110명) 때문인지, 아니면 이 코호트
규모 자체에서 오는 근본적인 평가 노이즈인지 구분하기 위한 대조군이다. 이 모델은
CNN/ViT/ABMIL이 전혀 없고 파라미터도 훨씬 적다 — 만약 이 모델도 seed마다 비슷한 폭으로
흔들린다면, 문제는 ViT 아키텍처가 아니라 표본 크기/평가방법론에 있다는 뜻이다.

train.py와 달리 WSI 처리가 전혀 없어 별도 스크립트로 뺐다(패치 forward/CNN/ViT/ABMIL/AMP
전부 불필요). wandb 로깅 없이 콘솔 출력만 한다 — 1회성 진단 실험용.

사용법:
    python train_clinical_rna_only.py --dataset tcga --seed 42
"""
import argparse
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, pdac_subtype_gene_ids
from models.clinical_encoder import ClinicalEncoder, age_stats_from_csv
from models.rna_encoder import RNAEncoder
from utils.losses import cox_ph_loss
from utils.metrics import compute_survival_metrics


class ClinicalRNAOnly(nn.Module):
    """age/sex + RNA-seq만으로 risk score를 만드는 WSI-free 모델."""

    def __init__(self, embed_dim: int, age_mean: float, age_std: float, rna_input_dim: int, dropout: float):
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(embed_dim, age_mean, age_std)
        self.rna_encoder = RNAEncoder(rna_input_dim, embed_dim, dropout=dropout)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 1),
        )

    def forward(self, age_years: torch.Tensor, sex_idx: torch.Tensor, rna: torch.Tensor) -> torch.Tensor:
        z_c = self.clinical_encoder(age_years.unsqueeze(0), sex_idx.unsqueeze(0)).squeeze(0)  # (D,)
        z_r = self.rna_encoder(rna.unsqueeze(0)).squeeze(0)                                    # (D,)
        fused = torch.cat([z_c, z_r], dim=-1)                                                  # (2D,)
        return self.risk_head(fused.unsqueeze(0)).view(1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _identity_collate(batch: list) -> list:
    return batch[0]


def _patient_risk(model: ClinicalRNAOnly, patient_slides: list, device) -> torch.Tensor:
    """슬라이드 리스트에서 환자 단위 메타데이터(age/sex/rna)만 뽑아 forward한다 — WSI 미사용."""
    p = patient_slides[0]
    age_years = p["age_years"].to(device)
    sex_idx   = p["sex_idx"].to(device)
    rna       = p["rna"].to(device)
    return model(age_years, sex_idx, rna)


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


def _log_line(prefix: str, metrics: dict) -> str:
    return (
        f"{prefix}_c_index={metrics['c_index']:.4f} | {prefix}_HR={metrics['hr']:.3f} "
        f"[{metrics['hr_ci_lower']:.3f}, {metrics['hr_ci_upper']:.3f}] | "
        f"{prefix}_logrank_p={metrics['log_rank_p']:.4f}"
    )


def _build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="cptac", choices=["tcga", "cptac"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--cox_batch_size", type=int, default=16)
    args = parser.parse_args()

    cfg = Config()
    cfg.data.seed = args.seed
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
    rna_input_dim = len(pdac_subtype_gene_ids())

    train_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="train", with_clinical=True, with_rna=True)
    val_ds   = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="val",   with_clinical=True, with_rna=True)
    test_ds  = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="test",  with_clinical=True, with_rna=True)

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    train_loader      = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **dl_kwargs)
    val_loader        = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader       = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    model = ClinicalRNAOnly(args.embed_dim, age_mean, age_std, rna_input_dim, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ClinicalRNAOnly (WSI 없음) | params={n_params:,} | rna_input_dim={rna_input_dim}")
    print(f"Dataset: {args.dataset}  seed={args.seed}  Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _build_scheduler(optimizer, args.epochs, warmup_epochs=3)

    best_score, best_state, best_epoch = -1.0, None, -1
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, args.cox_batch_size)
        train_metrics = evaluate(model, train_eval_loader, device)
        val_metrics   = evaluate(model, val_loader, device)
        scheduler.step()

        c_index = val_metrics.get("c_index", float("nan"))
        score = c_index if not math.isnan(c_index) else -1.0
        print(
            f"Epoch {epoch+1:3d} | loss={loss:.4f} | train_c_index={train_metrics['c_index']:.4f} | "
            + _log_line("val", val_metrics)
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    model.load_state_dict(best_state)
    train_metrics_final = evaluate(model, train_eval_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    print(f"\n=== Internal Test (best checkpoint epoch {best_epoch}) ===")
    print(_log_line("test", test_metrics))
    print(f"train_c_index(at best epoch)={train_metrics_final['c_index']:.4f}")


if __name__ == "__main__":
    main()
