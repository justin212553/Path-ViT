"""
CAMELYON17 패치 데이터셋 (wsi_train 기반)

stage_labels.csv 에서 환자 이름과 노드별 stage를 읽고,
wsi_train/<patient_name>/ 하위 패치를 로딩한다.

반환 형식:
    patches:     (N, 3, H, W)  float32
    coords:      (N, 2)        int64   [row, col]  (파일명 r####_c#### 파싱)
    label:       ()            int64   (0 = pN0/pN0(i+), 1 = pN1mi/pN1/pN2)
    patient_id:  str
    node_stages: dict[int, str]   {node_idx: stage}
"""
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import DataConfig

STAGE_TO_LABEL = {
    "pN0":     0,
    "pN0(i+)": 0,
    "pN1mi":   1,
    "pN1":     1,
    "pN2":     1,
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
        cfg:       DataConfig (wsi_root, test_root, csv_path, val_centers 참조)
        split:     "train" | "val" | "test"
        transform: 패치에 적용할 transform
    """

    def __init__(
        self,
        cfg: DataConfig,
        split: str = "train",
        transform=None,
    ):
        self.transform = transform or PATCH_TRANSFORM

        df = pd.read_csv(cfg.csv_path)

        # patient-level rows: patient_NNN.zip
        patient_df = df[df["patient"].str.endswith(".zip")].copy()
        patient_df["patient_id"] = patient_df["patient"].str.replace(".zip", "", regex=False)
        patient_df["center"] = (
            patient_df["patient_id"]
            .str.extract(r"patient_(\d+)")[0]
            .astype(int) // 20
        )
        patient_df["label"] = patient_df["stage"].map(STAGE_TO_LABEL)

        # node-level rows: patient_NNN_node_M.tif
        node_df = df[df["patient"].str.endswith(".tif")].copy()
        node_df["patient_id"] = node_df["patient"].str.extract(r"(patient_\d+)_node")[0]
        node_df["node"]       = node_df["patient"].str.extract(r"node_(\d+)")[0].astype(int)

        self._node_stages: dict[str, dict[int, str]] = {}
        for pid, grp in node_df.groupby("patient_id"):
            self._node_stages[pid] = dict(zip(grp["node"], grp["stage"]))

        if split == "test":
            self.wsi_root = Path(cfg.test_root)
            rows = patient_df
        else:
            self.wsi_root = Path(cfg.wsi_root)
            in_val = patient_df["center"].isin(cfg.val_centers)
            rows   = patient_df[in_val if split == "val" else ~in_val]

        # patches_train/ 과 patches_eval/ 모두 patient_XXX_node_Y/ flat 구조
        self.patients = rows[
            rows["patient_id"].apply(
                lambda p: any(self.wsi_root.glob(f"{p}_node_*"))
            )
        ].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        row        = self.patients.iloc[idx]
        patient_id = row["patient_id"]

        node_dirs   = sorted(self.wsi_root.glob(f"{patient_id}_node_*"))
        patch_paths = sorted(
            p for nd in node_dirs
            for p in list(nd.glob("*.png")) + list(nd.glob("*.jpg"))
        )

        patches_t = torch.stack([
            self.transform(Image.open(p).convert("RGB"))
            for p in patch_paths
        ])
        coords = torch.tensor(
            [_parse_coord(p.name) for p in patch_paths],
            dtype=torch.long,
        )

        return {
            "patches":     patches_t,
            "coords":      coords,
            "label":       torch.tensor(row["label"], dtype=torch.long),
            "patient_id":  patient_id,
            "node_stages": self._node_stages.get(patient_id, {}),
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
