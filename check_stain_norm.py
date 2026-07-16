"""
Stain normalization이 TCGA/CPTAC 배치 효과(check_domain_shift.py, raw CNN feature AUC 0.78)를
줄이는지 확인하는 소규모 검증 스크립트.

전체 코호트(~87K 패치)를 다시 뽑기 전에, 슬라이드 일부만 샘플링해 raw feature(이미 캐싱된
features.pt 재사용) vs Macenko stain-normalized feature로 도메인 분류기(TCGA=0/CPTAC=1)의
AUC를 비교한다. normalized 쪽 AUC가 raw보다 뚜렷이 낮아지면 전체 재추출을 검토할 근거가 된다.

target(정규화 기준 이미지)은 TCGA 첫 슬라이드의 첫 패치로 고정한다 — Macenko는 이 target의
염색 성분(H&E stain matrix)을 기준으로 다른 이미지를 정규화한다.

사용법:
    python check_stain_norm.py                       # 기본: 코호트당 슬라이드 15장
    python check_stain_norm.py --n-slides 20 --max-tiles-per-slide 40
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchstain
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torchvision import transforms

from config import DataConfig
from data.patch_utils import FEATURES_FILENAME, PATCH_TRANSFORM, list_patch_paths
from models.cnn_encoder import CNNEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STAIN_T = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)])


class DomainClassifier(nn.Module):
    """CNN feature(2048-dim) → 코호트(TCGA=0/CPTAC=1) 이진 분류. check_domain_shift.py와 동일 구조."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _sample_slide_dirs(patches_root: Path, n_slides: int, seed: int) -> list[Path]:
    tiles_root = patches_root / "tiles"
    dirs = sorted(d for d in tiles_root.iterdir() if d.is_dir() and (d / FEATURES_FILENAME).exists())
    rng = random.Random(seed)
    rng.shuffle(dirs)
    return dirs[:n_slides]


@torch.no_grad()
def _raw_and_normalized_features(
    encoder: CNNEncoder, normalizer, slide_dirs: list[Path], max_tiles_per_slide: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """슬라이드들에서 패치를 뽑아 (raw_features, normalized_features, slide_ids)를 나란히 만든다.

    raw는 이미 캐싱된 features.pt에서 해당 패치의 행만 골라 재사용하고(재계산 불필요),
    normalized는 Macenko 정규화 후 같은 frozen backbone으로 새로 forward한다.
    slide_ids는 이후 도메인 분류기 train/test split을 슬라이드 단위로 묶기 위한 group key —
    패치 단위로 무작위 분할하면 같은 슬라이드의 패치가 train/test에 동시에 섞여 슬라이드
    고유의 촬영/염색 "지문"을 암기하는 leakage가 생기고, 슬라이드 수가 적을수록(소규모 검증)
    이 leakage가 AUC를 천장(거의 1.0)까지 밀어올려 정규화 효과를 가린다.
    """
    raw_list, norm_list, slide_id_list = [], [], []
    for slide_idx, slide_dir in enumerate(slide_dirs):
        patch_paths = list_patch_paths(slide_dir)[:max_tiles_per_slide]
        if not patch_paths:
            continue
        cached = torch.load(slide_dir / FEATURES_FILENAME, weights_only=True)  # (N_full, 2048)
        raw_list.append(cached[: len(patch_paths)].numpy())
        slide_id_list.append(np.full(len(patch_paths), slide_idx))

        norm_batch = []
        for p in patch_paths:
            img = Image.open(p).convert("RGB")
            try:
                norm, _, _ = normalizer.normalize(I=STAIN_T(img), stains=True)  # (H,W,3) uint8-range
            except Exception:
                # Macenko가 조직/염색 성분을 못 찾는 극히 일부 패치(거의 배경 등)는 원본으로 대체
                norm_list_fallback = np.array(img)
                norm = torch.from_numpy(norm_list_fallback)
            norm_img = Image.fromarray(norm.clamp(0, 255).byte().cpu().numpy())
            norm_batch.append(PATCH_TRANSFORM(norm_img))
        norm_batch = torch.stack(norm_batch).to(DEVICE)
        feat_map = encoder.backbone(norm_batch)
        pooled = encoder.pool(feat_map)
        norm_list.append(pooled.cpu().numpy())

    return (
        np.concatenate(raw_list, axis=0),
        np.concatenate(norm_list, axis=0),
        np.concatenate(slide_id_list, axis=0),
    )


def _domain_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, epochs: int, lr: float) -> float:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    model = DomainClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = torch.from_numpy(X_train).float().to(DEVICE)
    y_train_t = torch.from_numpy(y_train).float().to(DEVICE)
    X_test_t  = torch.from_numpy(X_test).float().to(DEVICE)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(X_test_t)).cpu().numpy()
    return float(roc_auc_score(y_test, proba))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-slides", type=int, default=15, help="코호트당 샘플링할 슬라이드 수")
    parser.add_argument("--max-tiles-per-slide", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = DataConfig()
    tcga_dirs  = _sample_slide_dirs(Path(cfg.patches_root_tcga),  args.n_slides, args.seed)
    cptac_dirs = _sample_slide_dirs(Path(cfg.patches_root_cptac), args.n_slides, args.seed)
    print(f"TCGA slides: {len(tcga_dirs)}, CPTAC slides: {len(cptac_dirs)}")

    encoder = CNNEncoder(embed_dim=1, with_backbone=True).to(DEVICE)
    encoder.eval()
    encoder.requires_grad_(False)

    # target: TCGA 첫 슬라이드의 첫 패치를 정규화 기준으로 고정
    target_path = list_patch_paths(tcga_dirs[0])[0]
    target_img = Image.open(target_path).convert("RGB")
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    normalizer.fit(STAIN_T(target_img))
    print(f"Macenko target: {target_path}")

    print("Extracting TCGA raw+normalized features...")
    tcga_raw, tcga_norm, tcga_sid = _raw_and_normalized_features(
        encoder, normalizer, tcga_dirs, args.max_tiles_per_slide,
    )
    print(f"  TCGA patches: {len(tcga_raw)}")

    print("Extracting CPTAC raw+normalized features...")
    cptac_raw, cptac_norm, cptac_sid = _raw_and_normalized_features(
        encoder, normalizer, cptac_dirs, args.max_tiles_per_slide,
    )
    print(f"  CPTAC patches: {len(cptac_raw)}")

    X_raw  = np.concatenate([tcga_raw, cptac_raw], axis=0).astype(np.float32)
    X_norm = np.concatenate([tcga_norm, cptac_norm], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(len(tcga_raw)), np.ones(len(cptac_raw))]).astype(np.float32)
    # CPTAC slide id를 TCGA slide 수만큼 offset해서 두 코호트의 그룹 id가 겹치지 않게 한다
    groups = np.concatenate([tcga_sid, cptac_sid + tcga_sid.max() + 1])

    auc_raw  = _domain_auc(X_raw,  y, groups, args.seed, args.epochs, args.lr)
    auc_norm = _domain_auc(X_norm, y, groups, args.seed, args.epochs, args.lr)

    print("\n=== Domain shift check: raw vs Macenko-normalized ===")
    print(f"  Raw feature AUC        : {auc_raw:.4f}")
    print(f"  Normalized feature AUC : {auc_norm:.4f}")
    print(f"  변화: {auc_norm - auc_raw:+.4f}")


if __name__ == "__main__":
    main()
