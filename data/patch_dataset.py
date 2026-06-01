"""
CAMELYON17 패치 데이터셋 — 노드(슬라이드) 단위 MIL

각 아이템 = patient_NNN_node_M 슬라이드 하나의 패치 묶음 + 그 노드의 라벨.
분할은 환자 단위로 수행해 같은 환자의 노드가 train/val에 섞이지 않도록 한다.

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

import numpy as np
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


class CAMELYON17PatchDataset(Dataset):
    """
    Args:
        cfg:       DataConfig (wsi_root, test_root, csv_path, val_ratio 참조)
        split:     "train" | "val" | "test"
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

        df = pd.read_csv(cfg.csv_path)

        # 노드 단위 rows: patient_NNN_node_M.tif
        node_df = df[df["patient"].str.endswith(".tif")].copy()
        node_df["patient_id"] = node_df["patient"].str.extract(r"(patient_\d+)_node")[0]
        node_df["node"]       = node_df["patient"].str.extract(r"node_(\d+)")[0].astype(int)
        node_df["label"]      = node_df["stage"].map(STAGE_TO_LABEL)
        node_df = node_df.dropna(subset=["label"])   # 매핑 안 된 stage 제거
        node_df["label"] = node_df["label"].astype(int)

        if split == "test":
            self.wsi_root = Path(cfg.test_root)
            items_df = node_df
        else:
            self.wsi_root = Path(cfg.wsi_root)

            # 환자 단위 stratified split (data leakage 방지)
            patient_df = df[df["patient"].str.endswith(".zip")].copy()
            patient_df["patient_id"]    = patient_df["patient"].str.replace(".zip", "", regex=False)
            patient_df["patient_label"] = patient_df["stage"].map(STAGE_TO_LABEL)
            patient_df = patient_df.dropna(subset=["patient_label"])

            val_ratio = getattr(cfg, "val_ratio", 0.2)
            rng = np.random.default_rng(42)
            val_pids: set[str] = set()
            for lbl in patient_df["patient_label"].unique():
                grp = patient_df[patient_df["patient_label"] == lbl]["patient_id"].tolist()
                n_val = max(1, round(len(grp) * val_ratio))
                val_pids.update(rng.choice(grp, size=n_val, replace=False).tolist())

            in_val   = node_df["patient_id"].isin(val_pids)
            items_df = node_df[in_val if split == "val" else ~in_val]

        # 패치 파일이 1개 이상 존재하는 노드만 유지
        def _has_patches(r) -> bool:
            d = self.wsi_root / f"{r['patient_id']}_node_{r['node']}"
            return d.is_dir() and (next(d.glob("*.png"), None) or next(d.glob("*.jpg"), None)) is not None

        self.items = items_df[
            items_df.apply(_has_patches, axis=1)
        ].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        row      = self.items.iloc[idx]
        node_dir = self.wsi_root / f"{row['patient_id']}_node_{row['node']}"

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


class CAMELYON17NodeDataset(Dataset):
    """
    노드(WSI) 단위 데이터셋 — 히트맵 평가 전용.

    patches_eval/patch_index.csv 에서 패치별 GT 라벨을 읽어
    노드 하나를 하나의 아이템으로 반환한다.

    반환 형식:
        patches:      (N, 3, H, W)  float32
        coords:       (N, 2)        int64   [row, col]
        patch_labels: (N,)          int64   0=정상, 1=종양  (annotation 기반 GT)
        slide_id:     str           e.g. "patient_000_node_0"
    """

    def __init__(self, cfg: DataConfig, transform=None):
        self.transform = transform or PATCH_TRANSFORM
        self.root      = Path(cfg.test_root)

        index_df = pd.read_csv(self.root / "eval_patch_index.csv")

        self.slides = []
        for slide_id, grp in index_df.groupby("slide_id"):
            if (self.root / slide_id).is_dir():
                self.slides.append({
                    "slide_id": slide_id,
                    "df":       grp.reset_index(drop=True),
                })

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> dict:
        item     = self.slides[idx]
        slide_id = item["slide_id"]
        df       = item["df"]

        patches_t = torch.stack([
            self.transform(Image.open(Path(f)).convert("RGB"))
            for f in df["filename"]
        ])
        coords = torch.tensor(
            df[["row", "col"]].values,
            dtype=torch.long,
        )
        patch_labels = torch.tensor(
            df["patch_label"].values,
            dtype=torch.long,
        )

        return {
            "patches":      patches_t,
            "coords":       coords,
            "patch_labels": patch_labels,
            "slide_id":     slide_id,
        }
