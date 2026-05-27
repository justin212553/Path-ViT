"""
사전 추출된 JPEG 패치를 로딩하는 Dataset.

data/preprocess.py 로 생성된 출력 디렉토리를 사용한다.
OpenSlide 의존성 없음.

반환 형식:
    patches:      (N, 3, H, W)  float32
    coords:       (N, 2)        int64   [row, col]
    patch_labels: (N,)          int64   (-1 = unknown)
    label:        ()            int64
    slide_id:     str
    center_id:    int
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PATCH_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CAMELYON17PatchDataset(Dataset):
    """
    preprocess.py 로 생성된 JPEG 패치를 로딩한다.

    Args:
        preprocessed_root: slide_index.csv / patch_index.csv 가 있는 디렉토리
        split:             "train" | "val" | "test"
        val_centers:       validation 으로 사용할 center ID 튜플
        transform:         패치에 적용할 torchvision transform
    """

    def __init__(
        self,
        preprocessed_root: str,
        split: str = "train",
        val_centers: tuple = (1,),
        transform=None,
    ):
        self.transform = transform or PATCH_TRANSFORM
        root = Path(preprocessed_root)

        slide_index = pd.read_csv(root / "slide_index.csv")
        if split == "train":
            slide_index = slide_index[~slide_index["center_id"].isin(val_centers)]
        elif split == "val":
            slide_index = slide_index[slide_index["center_id"].isin(val_centers)]

        self.slides = slide_index.reset_index(drop=True)

        patch_index = pd.read_csv(root / "patch_index.csv")
        self.patch_groups = {
            sid: grp.reset_index(drop=True)
            for sid, grp in patch_index.groupby("slide_id")
        }

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> dict:
        slide = self.slides.iloc[idx]
        grp   = self.patch_groups[slide["slide_id"]]

        patches_t = torch.stack([
            self.transform(Image.open(row["filename"]).convert("RGB"))
            for _, row in grp.iterrows()
        ])

        return {
            "patches":      patches_t,
            "coords":       torch.tensor(grp[["row", "col"]].values, dtype=torch.long),
            "patch_labels": torch.tensor(grp["patch_label"].values,  dtype=torch.long),
            "label":        torch.tensor(slide["label"],              dtype=torch.long),
            "slide_id":     slide["slide_id"],
            "center_id":    int(slide["center_id"]),
        }
