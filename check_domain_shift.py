"""
Batch/domain shift 진단 스크립트 — CNN feature(2048-dim, data/extract_features.py가 만든
features.pt)만으로 "이 패치가 TCGA인지 CPTAC인지" 구분할 수 있는지 확인한다.

배경: train.py --dataset의 cross-dataset validation(TCGA↔CPTAC)에서 val c-index가
랜덤 수준(~0.45~0.47)에 머무는 문제를 진단하기 위한 도구다. CNN feature만으로 두 코호트를
쉽게(AUC가 1에 가깝게) 구분할 수 있다면, feature 공간에 기관/스캐너 특유의 배치 효과가
강하게 남아 있어 생존 신호보다 "어느 기관 슬라이드인가"가 더 쉽게 학습되고 있을 가능성이
크다는 뜻이다 — cross-dataset 전이 실패의 유력한 원인 후보.

사용법:
    python check_domain_shift.py
    python check_domain_shift.py --epochs 30 --hidden-dim 256
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from config import Config
from data.dataset import WSISurvivalDataset

BACKBONE_DIM = 2048  # CNNEncoder의 backbone 출력 차원 (models/cnn_encoder.py 참조)


class DomainClassifier(nn.Module):
    """CNN feature(2048-dim) → 코호트(TCGA=0 / CPTAC=1) 이진 분류. feature 공간의 linear probe 역할."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BACKBONE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 2048) → logit (N,)"""
        return self.net(x).squeeze(-1)


def _load_patch_features(cfg, dataset: str) -> np.ndarray:
    """dataset("tcga"|"cptac") 코호트 전체의 patch-level raw CNN feature (N, 2048)를 모은다."""
    ds = WSISurvivalDataset(cfg.data, dataset=dataset)
    feats = [
        slide["features"].numpy()
        for case_idx in range(len(ds))
        for slide in ds[case_idx]
    ]
    return np.concatenate(feats, axis=0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2, help="held-out 평가용 패치 비율")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = Config()
    cfg.data.precomputed = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading TCGA patch features...")
    tcga_feats = _load_patch_features(cfg, "tcga")
    print(f"  {tcga_feats.shape[0]:,} patches")

    print("Loading CPTAC patch features...")
    cptac_feats = _load_patch_features(cfg, "cptac")
    print(f"  {cptac_feats.shape[0]:,} patches")

    X = np.concatenate([tcga_feats, cptac_feats], axis=0).astype(np.float32)
    y = np.concatenate([
        np.zeros(len(tcga_feats), dtype=np.float32),
        np.ones(len(cptac_feats), dtype=np.float32),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed,
    )
    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    X_test  = torch.from_numpy(X_test).to(device)

    model     = DomainClassifier(args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn   = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss   = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1:3d} | loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(X_test)).cpu().numpy()
    auc = roc_auc_score(y_test, proba)

    n_tcga_test  = int((y_test == 0).sum())
    n_cptac_test = int((y_test == 1).sum())
    print("\n=== Domain shift check (TCGA vs CPTAC, CNN feature only) ===")
    print(f"  Test patches : {len(y_test):,}  (TCGA {n_tcga_test:,} / CPTAC {n_cptac_test:,})")
    print(f"  Test AUC     : {auc:.4f}")
    print("  AUC가 0.5에 가까움: CNN feature로 기관을 구분하기 어려움 (배치 효과 약함)")
    print("  AUC가 1.0에 가까움: CNN feature만으로 기관이 거의 완벽히 구분됨 (배치 효과 강함, "
          "cross-dataset 생존예측 실패의 유력한 원인 후보)")


if __name__ == "__main__":
    main()
