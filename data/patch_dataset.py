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


def _stratified_3way_split(
    all_slides: list,
    val_ratio: float,
    eval_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    """
    슬라이드를 pos(종양 패치 ≥1) / neg(전부 정상) 그룹별로 독립 3분할.
    각 split(train/val/eval)에 pos·neg 최소 1개씩 보장.

    Returns: (train_slides, val_slides, eval_slides)
    """
    rng = random.Random(seed)

    pos = [s for s in all_slides if int(s["df"]["patch_label"].sum()) > 0]
    neg = [s for s in all_slides if int(s["df"]["patch_label"].sum()) == 0]

    def _split_group(group: list) -> tuple[list, list, list]:
        g = list(group)
        rng.shuffle(g)
        n = len(g)
        n_eval  = max(1, round(n * eval_ratio))
        n_val   = max(1, round(n * val_ratio))
        n_train = n - n_val - n_eval
        if n_train < 1:
            # 슬라이드 수 부족 시 각 split에 1개씩 우선 배정
            n_train = max(1, n - 2)
            n_val   = 1 if n >= 2 else 0
            n_eval  = n - n_train - n_val
        return g[:n_train], g[n_train:n_train + n_val], g[n_train + n_val:n_train + n_val + n_eval]

    p_tr, p_va, p_ev = _split_group(pos)
    n_tr, n_va, n_ev = _split_group(neg)

    return p_tr + n_tr, p_va + n_va, p_ev + n_ev


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

        def _has_patches(r, root: Path) -> bool:
            d = root / f"{r['patient_id']}_node_{r['node']}"
            return d.is_dir() and (next(d.glob("*.png"), None) or next(d.glob("*.jpg"), None)) is not None

        if split == "test":
            self.wsi_root = Path(cfg.test_root)
            items_df = node_df

        elif split == "val":
            # wsi_eval(patches_eval)에서 positive 1개, negative 1개 랜덤 선택
            self.wsi_root = Path(cfg.test_root)
            eval_root = self.wsi_root
            has_eval = node_df.apply(lambda r: _has_patches(r, eval_root), axis=1)
            eval_df  = node_df[has_eval].reset_index(drop=True)

            pos_sample = eval_df[eval_df["label"] == 1].sample(5, random_state=42)
            neg_sample = eval_df[eval_df["label"] == 0].sample(5, random_state=42)
            items_df = pd.concat([pos_sample, neg_sample]).reset_index(drop=True)

        else:  # train: wsi_train(patches_train) 전체 사용
            self.wsi_root = Path(cfg.wsi_root)
            items_df = node_df

        # 패치 파일이 1개 이상 존재하는 노드만 유지
        train_root = self.wsi_root
        self.items = items_df[
            items_df.apply(lambda r: _has_patches(r, train_root), axis=1)
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
    노드(WSI) 단위 데이터셋 — 패치 레벨 GT 라벨 포함.

    patches_eval/eval_patch_index.csv 에서 패치별 GT 라벨을 읽어
    노드 하나를 하나의 아이템으로 반환한다.

    Args:
        split:       "all" | "train" | "val" | "eval"  (슬라이드 단위 stratified 3분할)
        val_ratio:   val 비율  (pos/neg 그룹 각각 적용, 기본 0.1)
        eval_ratio:  eval 비율 (pos/neg 그룹 각각 적용, 기본 0.1)
        seed:        분할 재현성 시드
        max_patches: 슬라이드당 최대 패치 수 (None=제한 없음, 학습 시 OOM 방지)

    반환 형식:
        patches:      (N, 3, H, W)  float32
        coords:       (N, 2)        int64   [row, col]
        patch_labels: (N,)          int64   0=정상, 1=종양  (annotation 기반 GT)
        slide_id:     str           e.g. "patient_000_node_0"
    """

    def __init__(
        self,
        cfg: DataConfig,
        split: str = "all",
        val_ratio: float = 0.1,
        eval_ratio: float = 0.1,
        seed: int = 42,
        max_patches: int | None = None,
        transform=None,
    ):
        self.transform   = transform or PATCH_TRANSFORM
        self.max_patches = max_patches
        self.root        = Path(cfg.test_root)

        index_df = pd.read_csv(self.root / "eval_patch_index.csv")

        all_slides = []
        for slide_id, grp in index_df.groupby("slide_id"):
            if (self.root / slide_id).is_dir():
                all_slides.append({
                    "slide_id": slide_id,
                    "df":       grp.reset_index(drop=True),
                })

        if split == "all":
            self.slides = all_slides
        else:
            train_slides, val_slides, eval_slides = _stratified_3way_split(
                all_slides, val_ratio, eval_ratio, seed
            )
            self.slides = {"train": train_slides, "val": val_slides, "eval": eval_slides}[split]

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> dict:
        item     = self.slides[idx]
        slide_id = item["slide_id"]
        df       = item["df"]

        if self.max_patches and len(df) > self.max_patches:
            df = df.sample(self.max_patches).reset_index(drop=True)

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
