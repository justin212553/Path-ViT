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
        cfg:       DataConfig (wsi_root, csv_path, val_centers 참조)
        split:     "train" | "val" | "test"
        transform: 패치에 적용할 transform
    """

    def __init__(
        self,
        cfg: DataConfig,
        split: str = "train",
        transform=None,
    ):
        if split == "test":
            # TODO: test split 구현
            raise NotImplementedError("test split is not implemented yet")

        self.transform = transform or PATCH_TRANSFORM
        self.wsi_root  = Path(cfg.wsi_root)

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

        # train/val split by center
        in_val = patient_df["center"].isin(cfg.val_centers)
        rows   = patient_df[in_val if split == "val" else ~in_val]

        # 실제 디렉토리가 존재하는 환자만 사용
        self.patients = rows[
            rows["patient_id"].apply(lambda p: (self.wsi_root / p).is_dir())
        ].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        row        = self.patients.iloc[idx]
        patient_id = row["patient_id"]
        patch_dir  = self.wsi_root / patient_id

        patch_paths = sorted(
            list(patch_dir.rglob("*.png")) + list(patch_dir.rglob("*.jpg"))
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
