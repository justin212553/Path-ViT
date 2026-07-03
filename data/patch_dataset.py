"""
CAMELYON16 패치 데이터셋 — 슬라이드 단위 MIL bag

patch_dataset.py(CAMELYON17)와 달리 CAMELYON16은 환자당 슬라이드가 1장이므로
"환자 단위로 여러 노드를 리스트로 묶는" 계층이 필요 없다. 아이템 단위 = 슬라이드 1장 = bag 1개.
__getitem__은 dict 하나를 그대로 반환한다 (patch_dataset.py처럼 List[dict]가 아님).

라벨 규칙:
    train: 슬라이드 파일명 접두사로 결정 (normal_* → 0, tumor_* → 1)
    test:  공식 reference.csv(image_name,label,Type)에서 매핑 (Normal → 0, Tumor → 1)
    ※ 지금은 train split만 다루므로 --manifest로 넘긴 csv가 없으면 파일명 규칙을 그대로 쓴다.

patches_root 하나에서 슬라이드 단위로 val(양성 최대 N / 음성 최대 N 랜덤)을 먼저 떼어내고
나머지 슬라이드 전부를 train으로 사용한다 (같은 슬라이드가 train/val에 동시에 들어가는
leakage는 애초에 발생하지 않음 — 슬라이드=bag이라 patient 단위 분리가 불필요).

DataLoader는 batch_size=1 + collate_fn=lambda batch: batch[0] 로 사용해야 한다
(bag마다 패치 수가 달라 배치로 stack할 수 없음. patch_dataset.py와 동일한 이유).

반환 형식 (dict 1개 = 슬라이드 1장):
    patch_paths: List[Path]   N개 패치 이미지 파일 경로 (지연 로딩, precomputed=False일 때만)
    features:    (N, D) float32 (precomputed=True일 때만, extract_features.py 산출물)
    coords:      (N, 2) int64 [row, col] (파일명 r####_c#### 파싱)
    label:       (1,) int64 (0=normal, 1=tumor)
    slide_id:    str
"""
import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import DataConfig

SEED = 42  # train/val 슬라이드 split 재현성
N_VAL_PER_CLASS = 10  # 클래스별 val로 뗄 슬라이드 수 (가용 슬라이드가 적으면 clamp)

PATCH_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

FEATURES_FILENAME = "features.pt"  # data/extract_features.py 산출물 파일명

_COORD_RE = re.compile(r"r(\d+)_c(\d+)")
_SLIDE_LABEL_RE = re.compile(r"^(normal|tumor|test)")


def _parse_coord(name: str) -> tuple[int, int]:
    m = _COORD_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def list_patch_paths(slide_dir: Path) -> list[Path]:
    """슬라이드 디렉터리의 패치 파일을 정렬된 순서로 나열.

    data/extract_features.py가 features.pt를 만들 때도 이 순서를 그대로 써야
    캐싱된 feature 행(row)과 패치(coords)가 어긋나지 않는다.
    """
    return sorted(list(slide_dir.glob("*.png")) + list(slide_dir.glob("*.jpg")))


def _label_from_filename(slide_id: str) -> int | None:
    """normal_XXX → 0, tumor_XXX → 1. test_XXX는 파일명만으로는 라벨을 알 수 없음(None)."""
    if slide_id.startswith("normal"):
        return 0
    if slide_id.startswith("tumor"):
        return 1
    return None


def _load_reference_labels(reference_csv: Path) -> dict:
    """공식 testing/reference.csv(image_name,label,Type) → {slide_id: 0/1} 매핑."""
    ref = pd.read_csv(reference_csv, header=None, names=["slide_id", "label", "type"])
    ref["label"] = ref["label"].str.strip().str.lower().map({"normal": 0, "tumor": 1})
    return dict(zip(ref["slide_id"], ref["label"]))


class CAMELYON16SlideDataset(Dataset):
    """
    Args:
        cfg: DataConfig (patches_root, precomputed 참조)
        split: "train" | "val" — patches_root 하나를 슬라이드 단위로 두 split으로 분할
        reference_csv: test 슬라이드 라벨이 필요할 때만 지정 (train split만 쓸 거면 None으로 둬도 됨)
        transform: 패치에 적용할 transform

    아이템 단위 = 슬라이드 1장 = bag 1개. __getitem__은 dict 하나를 반환한다.

    cfg.precomputed=True(기본값)면 data/extract_features.py로 미리 뽑아둔 features.pt를
    읽어 "features" 키로 반환하고, False면 패치 이미지 경로 리스트를 "patch_paths" 키로
    반환해 모델 forward에서 지연 디코딩하도록 한다.
    """

    def __init__(
        self,
        cfg: DataConfig,
        split: str = "train",
        reference_csv: Path | None = None,
        transform=None,
    ):
        self.transform = transform or PATCH_TRANSFORM
        self.root = Path(cfg.patches_root)
        self.precomputed = cfg.precomputed

        ref_labels = _load_reference_labels(reference_csv) if reference_csv else {}

        rows = []
        for slide_dir in sorted(self.root.iterdir()):
            if not slide_dir.is_dir():
                continue
            slide_id = slide_dir.name
            if not _SLIDE_LABEL_RE.match(slide_id):
                continue  # 패치 폴더가 아닌 항목(캐시 등) 무시

            label = _label_from_filename(slide_id)
            if label is None:
                label = ref_labels.get(slide_id)
            if label is None:
                continue  # 라벨을 못 찾은 슬라이드(예: reference_csv 없이 test_*) 제외

            rows.append({"slide_id": slide_id, "label": label})

        slide_df = pd.DataFrame(rows)

        # 패치 파일이 1개 이상 존재하는 슬라이드만 유지
        def _has_patches(r) -> bool:
            d = self.root / r["slide_id"]
            if self.precomputed:
                return (d / FEATURES_FILENAME).exists()
            return (next(d.glob("*.png"), None) or next(d.glob("*.jpg"), None)) is not None

        has_patches = slide_df.apply(_has_patches, axis=1)
        avail_df = slide_df[has_patches].reset_index(drop=True)

        # 양성/음성 슬라이드 각각에서 val로 N_VAL_PER_CLASS개 랜덤 샘플링
        # (가용 슬라이드가 N_VAL_PER_CLASS보다 적을 수 있으므로 실제 보유 수로 clamp)
        pos_group = avail_df[avail_df["label"] == 1]
        neg_group = avail_df[avail_df["label"] == 0]
        val_ids = set(
            pos_group.sample(min(N_VAL_PER_CLASS, len(pos_group)), random_state=SEED)["slide_id"]
        ) | set(
            neg_group.sample(min(N_VAL_PER_CLASS, len(neg_group)), random_state=SEED)["slide_id"]
        )
        is_val = avail_df["slide_id"].isin(val_ids)

        if split == "val":
            self.items = avail_df[is_val].reset_index(drop=True)
        else:  # train: val에 포함되지 않은 슬라이드 전체
            self.items = avail_df[~is_val].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        row = self.items.iloc[idx]
        slide_dir = self.root / row["slide_id"]
        patch_paths = list_patch_paths(slide_dir)

        coords = torch.tensor(
            [_parse_coord(p.name) for p in patch_paths],
            dtype=torch.long,
        )
        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()

        item = {
            "coords": coords,
            "label": torch.tensor([row["label"]], dtype=torch.long),
            "slide_id": row["slide_id"],
        }

        if self.precomputed:
            features = torch.load(slide_dir / FEATURES_FILENAME)
            if len(features) != len(patch_paths):
                raise RuntimeError(
                    f"{slide_dir}: features.pt 행 수({len(features)})가 패치 수"
                    f"({len(patch_paths)})와 다릅니다 — data/extract_features.py를 다시 실행하세요."
                )
            item["features"] = features
        else:
            item["patch_paths"] = patch_paths

        return item
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
