"""
CAMELYON17 패치 데이터셋 — 노드(슬라이드) 단위 MIL

각 아이템 = 환자 1명의 모든 노드(슬라이드) 리스트. 노드별 라벨/예측 단위는 그대로 유지하되,
DataLoader가 한 번에 꺼내는 단위(배치/gradient accumulation 단위)를 환자로 묶기 위함이다.
patches_root 하나에서 환자 단위로 val(양성 환자 최대 10 / 음성 환자 최대 10 랜덤)을 먼저 떼어내고
나머지 환자의 노드 전부를 train으로 사용한다 (같은 환자가 train/val에 동시에 들어가는
leakage 방지, 모델 파이프라인 점검용 — eval split 없음).

DataLoader는 batch_size=1 + collate_fn=lambda batch: batch[0] 로 사용해야 한다.

반환 형식 (환자 1명의 노드 수만큼의 리스트, 각 원소는 dict):
    patch_paths: List[Path]  N개   패치 이미지 파일 경로 (이미지 디코딩은 모델 forward에서
                                   chunk_size 단위로 지연 로딩 — 패치 수가 매우 큰 WSI에서
                                   전체를 한 번에 메모리에 올려 OOM 나는 것을 방지)
    coords:      (N, 2)       int64   [row, col]  (파일명 r####_c#### 파싱)
    label:       (1,)         int64   (0=음성, 1=전이)
    patient_id:  str
    node:        int
"""
import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import DataConfig

SEED = 42  # train/val 환자 split 재현성

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

FEATURES_FILENAME = "features.pt"  # data/extract_features.py 산출물 파일명

_COORD_RE = re.compile(r"r(\d+)_c(\d+)")


def _parse_coord(name: str) -> tuple[int, int]:
    m = _COORD_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def list_patch_paths(node_dir: Path) -> list[Path]:
    """노드 디렉터리의 패치 파일을 정렬된 순서로 나열.

    data/extract_features.py가 features.pt를 만들 때도 이 순서를 그대로 써야
    캐싱된 feature 행(row)과 패치(coords)가 어긋나지 않는다.
    """
    return sorted(list(node_dir.glob("*.png")) + list(node_dir.glob("*.jpg")))


class CAMELYON17NodeDataset(Dataset):
    """
    Args:
        cfg:       DataConfig (patches_root, csv_path, precomputed 참조)
        split:     "train" | "val" — patches_root 하나를 환자 단위로 두 split으로 분할
        transform: 패치에 적용할 transform

    아이템 단위 = 환자 1명. __getitem__은 그 환자가 가진 모든 노드의 dict 리스트를 반환한다.
    cfg.precomputed=True(기본값)면 data/extract_features.py로 미리 뽑아둔 features.pt를
    읽어 "features" 키로 반환하고, False면 패치 이미지 경로 리스트를 "patch_paths" 키로
    반환해 모델 forward에서 지연 디코딩하도록 한다.
    """

    def __init__(
        self,
        cfg: DataConfig,
        split: str = "train",
        transform=None,
    ):
        self.transform    = transform or PATCH_TRANSFORM
        self.root         = Path(cfg.patches_root)
        self.precomputed  = cfg.precomputed

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
            if not d.is_dir():
                return False
            if self.precomputed:
                return (d / FEATURES_FILENAME).exists()
            return (next(d.glob("*.png"), None) or next(d.glob("*.jpg"), None)) is not None

        # 패치 파일이 1개 이상 존재하는 노드만 유지
        has_patches = node_df.apply(_has_patches, axis=1)
        avail_df    = node_df[has_patches].reset_index(drop=True)

        # 환자 단위로 양성(노드 중 1개라도 전이) / 음성(전부 정상) 분류 후 val 환자 샘플링
        # (가용 환자가 10명보다 적을 수 있으므로 — 예: 패치 재추출 중간 — 실제 보유 수로 clamp)
        patient_label  = avail_df.groupby("patient_id")["label"].max()
        pos_group      = patient_label[patient_label == 1]
        neg_group      = patient_label[patient_label == 0]
        pos_patients   = pos_group.sample(min(10, len(pos_group)), random_state=SEED).index
        neg_patients   = neg_group.sample(min(10, len(neg_group)), random_state=SEED).index
        val_patients   = set(pos_patients) | set(neg_patients)

        is_val = avail_df["patient_id"].isin(val_patients)
        if split == "val":
            self.items = avail_df[is_val].reset_index(drop=True)
        else:  # train: val에 포함되지 않은 환자의 노드 전체
            self.items = avail_df[~is_val].reset_index(drop=True)

        self.patients = sorted(self.items["patient_id"].unique())

    def __len__(self) -> int:
        return len(self.patients)

    def _load_node(self, row) -> dict:
        node_dir = self.root / f"{row['patient_id']}_node_{row['node']}"

        patch_paths = list_patch_paths(node_dir)

        coords = torch.tensor(
            [_parse_coord(p.name) for p in patch_paths],
            dtype=torch.long,
        )
        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()

        item = {
            "coords":     coords,
            "label":      torch.tensor([row["label"]], dtype=torch.long),
            "patient_id": row["patient_id"],
            "node":       int(row["node"]),
        }

        if self.precomputed:
            features = torch.load(node_dir / FEATURES_FILENAME)
            if len(features) != len(patch_paths):
                raise RuntimeError(
                    f"{node_dir}: features.pt 행 수({len(features)})가 패치 수"
                    f"({len(patch_paths)})와 다릅니다 — data/extract_features.py를 다시 실행하세요."
                )
            item["features"] = features
        else:
            item["patch_paths"] = patch_paths

        return item

    def __getitem__(self, idx: int) -> list:
        patient_id   = self.patients[idx]
        patient_rows = self.items[self.items["patient_id"] == patient_id]
        return [self._load_node(row) for _, row in patient_rows.iterrows()]
