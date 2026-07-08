"""
TCGA-PAAD / CPTAC-PDA WSI 생존(OS) 데이터셋 — 환자(case) 단위 MIL.

data/patch_dataset.py(CAMELYON17NodeDataset — 환자당 다중 노드를 리스트로 묶는 구조)를
그대로 참고했다. 다만 라벨이 이진 분류(stage)가 아니라 OS_time/OS_event(생존 분석) 이고,
슬라이드→환자 매핑도 파일명 파싱이 아니라 data/preprocess_cptac.py 산출물인
slide_index_task*.csv의 case_id 컬럼을 그대로 쓴다는 점이 다르다.

각 아이템 = 환자(case) 1명이 보유한 모든 슬라이드 리스트(dict). CAMELYON17과 동일하게
DataLoader는 batch_size=1 + collate_fn=lambda batch: batch[0] 로 사용해야 한다.

반환 형식 (환자 1명의 슬라이드 수만큼의 리스트, 각 원소는 dict):
    patch_paths / features: patch_dataset.py와 동일 — precomputed 여부에 따라 둘 중 하나만 존재
    coords:      (N, 2) int64   [row, col]  (파일명 r####_c#### 파싱)
    case_id:     str
    slide_id:    str
    dataset:     "tcga" | "cptac"
    OS_time:     (1,) float32
    OS_event:    (1,) int64   (1=사망, 0=생존/censored)

data/extract_os_labels.py 산출물(data/os_labels_{tcga,cptac}.csv)에 없는 case(=raw clinical.tsv에
없거나 vital_status 미상이라 OS를 알 수 없는 환자)의 슬라이드는 라벨이 없으므로 제외한다.

train/val split은 OS_event(사망/생존) 그룹별로 환자 단위 val 샘플링을 한다
(patch_dataset.py의 양성/음성 10명씩 val 규칙과 동일한 취지 — 슬라이드가 아니라 환자 단위로 나눠야
같은 환자의 슬라이드가 train/val에 동시에 들어가는 leakage를 막을 수 있다).

사용법 예:
    from config import DataConfig
    from data.dataset import WSISurvivalDataset
    train_ds = WSISurvivalDataset(DataConfig(), dataset="cptac", split="train")
"""
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DataConfig
from data.patch_dataset import FEATURES_FILENAME, PATCH_TRANSFORM, list_patch_paths, _parse_coord

SEED = 42
VAL_PER_GROUP = 15  # OS_event(사망/생존) 그룹별 val 환자 수

OS_LABEL_PATHS = {
    "tcga":  Path("data/os_labels_tcga.csv"),
    "cptac": Path("data/os_labels_cptac.csv"),
}
PATCHES_ROOT_ATTRS = {
    "tcga":  "patches_root_tcga",
    "cptac": "patches_root_cptac",
}


def _load_slide_index(patches_root: Path) -> pd.DataFrame:
    """data/preprocess_cptac.py가 --num-tasks 샤드별로 나눠 쓴 slide_index_task*.csv를 모두 합친다."""
    paths = sorted(patches_root.glob("slide_index_task*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"{patches_root}에 slide_index_task*.csv가 없습니다 — "
            "먼저 python -m data.preprocess_cptac 을 실행하세요."
        )
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


class WSISurvivalDataset(Dataset):
    """
    Args:
        cfg:       DataConfig (patches_root_tcga/cptac, precomputed 참조)
        dataset:   "tcga" | "cptac"
        split:     "train" | "val"
        transform: 패치에 적용할 transform (precomputed=False일 때만 사용)

    아이템 단위 = 환자 1명. __getitem__은 그 환자가 가진 모든 슬라이드의 dict 리스트를 반환한다.
    """

    def __init__(
        self,
        cfg: DataConfig,
        dataset: str = "cptac",
        split: str = "train",
        transform=None,
    ):
        if dataset not in OS_LABEL_PATHS:
            raise ValueError(f"dataset must be one of {list(OS_LABEL_PATHS)}, got {dataset!r}")

        self.dataset     = dataset
        self.transform   = transform or PATCH_TRANSFORM
        self.root        = Path(getattr(cfg, PATCHES_ROOT_ATTRS[dataset]))
        self.precomputed = cfg.precomputed

        slide_df = _load_slide_index(self.root)
        slide_df = slide_df[(slide_df["status"] == "ok") & (slide_df["n_tiles_kept"] > 0)]

        os_df  = pd.read_csv(OS_LABEL_PATHS[dataset])
        merged = slide_df.merge(os_df[["case_id", "OS_time", "OS_event"]], on="case_id", how="inner")

        def _has_patches(slide_id: str) -> bool:
            d = self.root / "tiles" / slide_id
            if self.precomputed:
                return (d / FEATURES_FILENAME).exists()
            return (next(d.glob("*.jpg"), None) or next(d.glob("*.png"), None)) is not None

        has_patches = merged["slide_id"].apply(_has_patches)
        avail_df    = merged[has_patches].reset_index(drop=True)
        if avail_df.empty:
            raise RuntimeError(
                f"[{dataset}] 사용 가능한 슬라이드가 없습니다 — preprocess_cptac 산출물과 "
                f"os_labels 병합 결과를 확인하세요."
            )

        # 환자(case) 단위 OS_event 그룹별로 val 환자 샘플링 (사망/생존 분포를 val/train에 고르게 유지)
        case_event = avail_df.groupby("case_id")["OS_event"].first()
        val_cases  = set()
        for _, group in case_event.groupby(case_event):
            n = min(VAL_PER_GROUP, len(group))
            val_cases |= set(group.sample(n, random_state=SEED).index)

        is_val = avail_df["case_id"].isin(val_cases)
        self.items = (avail_df[is_val] if split == "val" else avail_df[~is_val]).reset_index(drop=True)

        self.cases = sorted(self.items["case_id"].unique())

    def __len__(self) -> int:
        return len(self.cases)

    def _load_slide(self, row) -> dict:
        slide_dir   = self.root / "tiles" / row["slide_id"]
        patch_paths = list_patch_paths(slide_dir)

        coords = torch.tensor(
            [_parse_coord(p.name) for p in patch_paths],
            dtype=torch.long,
        )
        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()

        item = {
            "coords":   coords,
            "case_id":  row["case_id"],
            "slide_id": row["slide_id"],
            "dataset":  self.dataset,
            "OS_time":  torch.tensor([row["OS_time"]], dtype=torch.float32),
            "OS_event": torch.tensor([row["OS_event"]], dtype=torch.long),
        }

        if self.precomputed:
            features = torch.load(slide_dir / FEATURES_FILENAME)
            if len(features) != len(patch_paths):
                raise RuntimeError(
                    f"{slide_dir}: features.pt 행 수({len(features)})가 패치 수"
                    f"({len(patch_paths)})와 다릅니다 — utils.extract_features를 다시 실행하세요."
                )
            item["features"] = features
        else:
            item["patch_paths"] = patch_paths

        return item

    def __getitem__(self, idx: int) -> list:
        case_id   = self.cases[idx]
        case_rows = self.items[self.items["case_id"] == case_id]
        return [self._load_slide(row) for _, row in case_rows.iterrows()]
