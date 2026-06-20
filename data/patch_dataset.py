"""
CAMELYON17 패치 데이터셋 — 노드(슬라이드) 단위 MIL

각 아이템 = patient_NNN_node_M 슬라이드 하나의 패치 묶음 + 그 노드의 라벨.
patches_root 하나에서 val(positive 5 / negative 5 랜덤)을 먼저 떼어내고
나머지 전부를 train으로 사용한다 (모델 파이프라인 점검용 — eval split 없음).

반환 형식:
    patches:    (N, 3, H, W)  float32
    coords:     (N, 2)        int64   [row, col]  (파일명 r####_c#### 파싱)
    label:      ()            int64   (0=음성, 1=전이)
    patient_id: str
    node:       int
"""
import random
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import DataConfig

STAGE_TO_LABEL = {
    "pN0":      0,
    "pN0(i+)":  0,
    "negative": 0,
    "itc":      0,
    "pN1mi":    1,
    "micro":    1,
    "pN1":      1,
    "macro":    1,
    "pN2":      1,
}

PATCH_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_COORD_RE = re.compile(r"r(\d+)_c(\d+)")


def _parse_coord(name: str) -> tuple[int, int]:
    m = _COORD_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


class CAMELYON17NodeDataset(Dataset):
    """
    Args:
        cfg:       DataConfig (patches_root, csv_path 참조)
        split:     "train" | "val" — patches_root 하나를 두 split으로 분할
        transform: 패치에 적용할 transform
    """

    def __init__(
        self,
        cfg: DataConfig,
        split: str = "train",
        transform=None,
        max_patches: int = 2000,
    ):
        self.transform   = transform or PATCH_TRANSFORM
        self.max_patches = max_patches
        self.root        = Path(cfg.patches_root)

        df = pd.read_csv(cfg.csv_path)

        # 노드 단위 rows: patient_NNN_node_M.tif
        node_df = df[df["patient"].str.endswith(".tif")].copy()
        node_df["patient_id"] = node_df["patient"].str.extract(r"(patient_\d+)_node")[0]
        node_df["node"]       = node_df["patient"].str.extract(r"node_(\d+)")[0].astype(int)
        node_df["label"]      = node_df["stage"].map(STAGE_TO_LABEL)
        node_df = node_df.dropna(subset=["label"])   # 매핑 안 된 stage 제거
        node_df["label"] = node_df["label"].astype(int)

        def _has_patches(r) -> bool:
            d = self.root / f"{r['patient_id']}_node_{r['node']}"
            return d.is_dir() and (next(d.glob("*.png"), None) or next(d.glob("*.jpg"), None)) is not None

        # 패치 파일이 1개 이상 존재하는 노드만 유지
        has_patches = node_df.apply(_has_patches, axis=1)
        avail_df    = node_df[has_patches].reset_index(drop=True)

        # val: positive 5개, negative 5개 랜덤 선택 / train: 나머지 전체
        pos_sample = avail_df[avail_df["label"] == 1].sample(5, random_state=42)
        neg_sample = avail_df[avail_df["label"] == 0].sample(5, random_state=42)
        val_idx    = pos_sample.index.union(neg_sample.index)

        if split == "val":
            self.items = pd.concat([pos_sample, neg_sample]).reset_index(drop=True)
        else:  # train
            self.items = avail_df.drop(val_idx).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        row      = self.items.iloc[idx]
        node_dir = self.root / f"{row['patient_id']}_node_{row['node']}"

        patch_paths = sorted(
            list(node_dir.glob("*.png")) + list(node_dir.glob("*.jpg"))
        )

        if self.max_patches and len(patch_paths) > self.max_patches:
            patch_paths = random.sample(patch_paths, self.max_patches)

        patches_t = torch.stack([
            self.transform(Image.open(p).convert("RGB"))
            for p in patch_paths
        ])
        coords = torch.tensor(
            [_parse_coord(p.name) for p in patch_paths],
            dtype=torch.long,
        )
        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()

        return {
            "patches":    patches_t,
            "coords":     coords,
            "label":      torch.tensor(row["label"], dtype=torch.long),
            "patient_id": row["patient_id"],
            "node":       int(row["node"]),
        }
