"""
TCGA-BRCA(1061 case, UNI backbone feature) internal 전용 학습/평가 스크립트.

`scripts/prepare_brca_data.py`가 만든 데이터(슬라이드당 이미 pooled된 UNI feature .pt + 좌표
.pt, 이미지 파일 없음)는 `data/dataset.py::WSISurvivalDataset`(패치 "이미지 파일"이 있고 좌표를
파일명에서 파싱하는 구조)와 형식이 안 맞아 별도 로더를 새로 짠다.

RNA-seq 데이터가 없다(BRCA는 WSI feature + 임상만 준비됨) — 그래서 M7(RNA+Clinical)/
PMA_EX_SS_AUX(WSI+RNA+Clinical) 그대로는 못 돌리고, 대신:
  --model clinical_only : Clinical(age/sex)만, WSI 없음 — M5과 동일한 스펙의 WSI-free 대조군
  --model wsi_clinical   : WSI(ABMIL gated-attention pooling) + Clinical — M2와 동일한 스펙
                           (Late Fusion), RNA-guided 부분만 없음
을 비교한다. "표본을 1000명대(TCGA-PAAD 152명 -> BRCA 1061명)로 늘리면 WSI가 Clinical 대비
순증분 기여를 하는가"가 핵심 질문.

ViT/Nystromformer 공간 컨텍스트 블록은 이번엔 안 쓴다 — 슬라이드당 패치 수가 중앙값 10,310개로
PAAD(중앙값 67개)와 규모가 완전히 다르고, 이 실험의 범위는 "WSI 정보 자체의 유무"이지 "공간
정보의 정교함"이 아니다(공간 블록 자체의 가치는 PAAD 쪽에서 별도 검증 중).

TCGA 단일 코호트라 external(cross-institution) 검증은 불가능 — internal(held-out test)만 본다.

사용법:
    python -m scripts.train_brca --model clinical_only --seed 42
    python -m scripts.train_brca --model wsi_clinical --seed 42
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.clinical_encoder import ClinicalEncoder, encode_sex
from models.uni_encoder import UNIEncoder
from models.vit_m1 import AttentionPooling
from utils.losses import cox_ph_loss
from utils.metrics import compute_survival_metrics

MANIFEST_PATH = Path("data/brca_slide_manifest.csv")
TILES_ROOT = Path("data/patches_tcga_brca/tiles")
EMBED_DIM = 64  # 우리 프로젝트 표준(cfg.model.embed_dim)과 동일하게 맞춤
DROPOUT = 0.3


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_cases() -> list[dict]:
    manifest = pd.read_csv(MANIFEST_PATH)
    cases = []
    for case_id, group in manifest.groupby("case_id"):
        row = group.iloc[0]
        cases.append({
            "case_id": case_id,
            "slide_ids": group["slide_id"].tolist(),
            "OS_time": float(row["OS_time"]),
            "OS_event": int(row["OS_event"]),
            "age_years": float(row["age_years"]),
            "sex": row["sex"],
        })
    return cases


def _split_cases(cases: list[dict], seed: int) -> tuple[list, list, list]:
    events = [c["OS_event"] for c in cases]
    train_valid, test = train_test_split(cases, test_size=0.2, random_state=seed, stratify=events)
    events_tv = [c["OS_event"] for c in train_valid]
    train, valid = train_test_split(train_valid, test_size=0.25, random_state=seed, stratify=events_tv)
    return train, valid, test


class WSIClinicalModel(nn.Module):
    """WSI(UNI feature -> proj -> ABMIL attention pooling, 슬라이드 평균) + Clinical(age/sex),
    late-fusion concat -> risk_head. models/vit_m2.py와 동일한 설계 사상이지만 ViT/Nystromformer
    공간 컨텍스트 블록 없이 순수 MIL pooling만 쓴다(모듈 docstring 참조)."""

    def __init__(self, age_mean: float, age_std: float):
        super().__init__()
        self.uni = UNIEncoder(EMBED_DIM, with_backbone=False)
        self.attn_pool = AttentionPooling(EMBED_DIM, hidden_dim=128)
        self.clinical_encoder = ClinicalEncoder(EMBED_DIM, age_mean, age_std)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM * 2),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED_DIM * 2, 1),
        )

    def forward(self, slide_features: list[torch.Tensor], age_years: torch.Tensor, sex_idx: torch.Tensor) -> torch.Tensor:
        slide_embeds = []
        for features in slide_features:
            tokens = self.uni.forward_pooled(features)      # (N, embed_dim)
            wsi_embed, _ = self.attn_pool(tokens)            # (embed_dim,)
            slide_embeds.append(wsi_embed)
        z_wsi = torch.stack(slide_embeds).mean(dim=0)         # (embed_dim,) 슬라이드 평균 풀링
        z_clinical = self.clinical_encoder(age_years.unsqueeze(0), sex_idx.unsqueeze(0)).squeeze(0)
        combined = torch.cat([z_wsi, z_clinical], dim=-1)
        return self.risk_head(combined.unsqueeze(0)).view(1)


class ClinicalOnlyModel(nn.Module):
    """M5과 동일한 스펙: age/sex만 -> risk_head. WSI-free 대조군."""

    def __init__(self, age_mean: float, age_std: float):
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(EMBED_DIM, age_mean, age_std)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED_DIM, 1),
        )

    def forward(self, slide_features, age_years: torch.Tensor, sex_idx: torch.Tensor) -> torch.Tensor:
        z_clinical = self.clinical_encoder(age_years.unsqueeze(0), sex_idx.unsqueeze(0)).squeeze(0)
        return self.risk_head(z_clinical.unsqueeze(0)).view(1)


def _load_slide_features(case: dict, device) -> list[torch.Tensor] | None:
    feats = []
    for slide_id in case["slide_ids"]:
        p = TILES_ROOT / slide_id / "features_uni.pt"
        if not p.exists():
            continue
        feats.append(torch.load(p, weights_only=True).to(device))
    return feats or None


def _patient_risk(model, case: dict, device, needs_wsi: bool) -> torch.Tensor:
    age_years = torch.tensor(case["age_years"], device=device)
    sex_idx = encode_sex([case["sex"]]).to(device).squeeze(0)
    slide_features = _load_slide_features(case, device) if needs_wsi else None
    return model(slide_features, age_years, sex_idx)


@torch.no_grad()
def _evaluate(model, cases: list[dict], device, needs_wsi: bool) -> dict:
    model.eval()
    risks, times, events = [], [], []
    for case in cases:
        if needs_wsi and _load_slide_features(case, device) is None:
            continue
        risk = _patient_risk(model, case, device, needs_wsi)
        risks.append(risk.item())
        times.append(case["OS_time"])
        events.append(case["OS_event"])
    return compute_survival_metrics(np.array(risks), np.array(times), np.array(events))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, choices=["clinical_only", "wsi_clinical"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None,
                         help="기본: clinical_only=1e-3(train_light.py 관례), wsi_clinical=1e-5(WSI 모델 관례)")
    parser.add_argument("--weight-decay", type=float, default=1e-1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    needs_wsi = args.model == "wsi_clinical"
    lr = args.lr or (1e-5 if needs_wsi else 1e-3)

    cases = _load_cases()
    print(f"전체 case 수: {len(cases)}")
    train, val, test = _split_cases(cases, args.seed)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    age_mean = float(np.mean([c["age_years"] for c in train]))
    age_std = float(np.std([c["age_years"] for c in train], ddof=0))
    print(f"train age_mean={age_mean:.2f} age_std={age_std:.2f}")

    model = (WSIClinicalModel if needs_wsi else ClinicalOnlyModel)(age_mean, age_std).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  params={n_params:,}  lr={lr:.1e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7)

    best_val_c, best_state, epochs_since_improve = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(train))
        risks, times, events = [], [], []
        total_loss, n_batches = 0.0, 0

        def _flush():
            nonlocal risks, times, events, total_loss, n_batches
            if not risks:
                return
            risk_t = torch.cat(risks)
            time_t = torch.tensor(times, device=device)
            event_t = torch.tensor(events, device=device)
            loss = cox_ph_loss(risk_t, time_t, event_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            risks, times, events = [], [], []

        for i in perm:
            case = train[i]
            if needs_wsi and _load_slide_features(case, device) is None:
                continue
            risk = _patient_risk(model, case, device, needs_wsi)
            risks.append(risk)
            times.append(case["OS_time"])
            events.append(case["OS_event"])
            if len(risks) >= args.batch_size:
                _flush()
        _flush()

        val_metrics = _evaluate(model, val, device, needs_wsi)
        val_c = val_metrics["c_index"]
        score = val_c if not np.isnan(val_c) else -1.0
        scheduler.step(score)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d} | lr={lr_now:.2e} | loss={total_loss/max(n_batches,1):.4f} | val_c_index={val_c:.4f}")

        if score > best_val_c:
            best_val_c = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"early stopping at epoch {epoch} (best val_c_index={best_val_c:.4f})")
                break

    model.load_state_dict(best_state)
    test_metrics = _evaluate(model, test, device, needs_wsi)

    print(f"\n=== RESULT (model={args.model}, seed={args.seed}) ===")
    print(f"best_val_c_index={best_val_c:.4f}")
    print(f"test_c_index={test_metrics['c_index']:.4f} | test_HR={test_metrics['hr']:.3f} "
          f"[{test_metrics['hr_ci_lower']:.3f}, {test_metrics['hr_ci_upper']:.3f}] | "
          f"test_logrank_p={test_metrics['log_rank_p']:.4f}")


if __name__ == "__main__":
    main()
