"""
PathVitData — Path-ViT의 raw patch 데이터셋(CAMELYON17NodeDataset)을 TransMIL이 사용할 수
있게 연결하는 adapter.

TransMIL(CamelData)은 미리 추출된 (N_patches, 1024) feature bag(.pt)을 읽는 반면, Path-ViT의
CAMELYON17NodeDataset은 패치 이미지 경로 리스트를 반환한다. 여기서는 노드(슬라이드) 단위로
펼친 뒤, 고정(freeze)된 CNNEncoder(ResNet50)로 패치를 인코딩해 동일한 (features, label) 포맷을
맞춰 TransMIL 파이프라인(datasets/data_interface.py, models/model_interface.py)을 그대로 쓸 수
있게 한다. train/val split은 Path-ViT train.py와 동일한 환자 단위 split(SEED=42)을 재사용한다.

주의: CNN 인코딩을 __getitem__ 내부에서 GPU로 수행하므로, DataLoader는 num_workers=0으로
사용해야 한다 (그 외에는 CUDA가 이미 초기화된 프로세스를 fork하면서 오류 발생).
"""
import importlib.util
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

_ROOT = Path(__file__).resolve().parents[2]  # Path-ViT 루트
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from config import DataConfig                                    # noqa: E402
from data.patch_dataset import CAMELYON17NodeDataset               # noqa: E402


def _load_cnn_encoder_cls():
    # plain `import models.cnn_encoder`는 TransMIL 자체의 `models` 패키지와 이름이
    # 충돌하므로, 파일 경로로 직접 로드해 별도 모듈 네임스페이스로 분리한다.
    spec = importlib.util.spec_from_file_location(
        "pathvit_cnn_encoder", _ROOT / "models" / "cnn_encoder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CNNEncoder


_CNNEncoder = _load_cnn_encoder_cls()


class PathVitData(Dataset):
    """
    각 아이템 = WSI 노드(슬라이드) 1개.
    Returns:
        features: (N_patches, feat_dim) float32 텐서
        label:    int (0=음성, 1=전이)
    """

    def __init__(self, dataset_cfg=None, state=None):
        self.dataset_cfg = dataset_cfg

        patches_root = Path(dataset_cfg.patches_root or "data/patches")
        csv_path     = Path(dataset_cfg.csv_path or "data/stage_labels.csv")
        if not patches_root.is_absolute():
            patches_root = _ROOT / patches_root
        if not csv_path.is_absolute():
            csv_path = _ROOT / csv_path

        self.feat_dim    = dataset_cfg.feat_dim or 1024
        self.chunk_size  = dataset_cfg.chunk_size or 64
        self.shuffle     = bool(dataset_cfg.data_shuffle)

        # Path-ViT에는 별도 test split이 없으므로 test 단계는 val split을 재사용한다.
        split = "train" if state == "train" else "val"

        node_cfg = DataConfig(patches_root=str(patches_root), csv_path=str(csv_path))
        node_ds  = CAMELYON17NodeDataset(node_cfg, split=split)

        self.items     = node_ds.items.reset_index(drop=True)  # 노드 단위 rows
        self.transform = node_ds.transform
        self._load_node = node_ds._load_node

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = _CNNEncoder(embed_dim=self.feat_dim, pretrained=True).to(self.device).eval()
        self.encoder.requires_grad_(False)

    def __len__(self) -> int:
        return len(self.items)

    @torch.no_grad()
    def __getitem__(self, idx: int):
        row  = self.items.iloc[idx]
        node = self._load_node(row)
        patch_paths = node["patch_paths"]
        label       = int(node["label"].item())

        feats = []
        for i in range(0, len(patch_paths), self.chunk_size):
            chunk = patch_paths[i : i + self.chunk_size]
            imgs = torch.stack([
                self.transform(Image.open(p).convert("RGB")) for p in chunk
            ]).to(self.device, non_blocking=True)
            feats.append(self.encoder(imgs).cpu())
        features = torch.cat(feats, dim=0)

        if self.shuffle:
            features = features[torch.randperm(features.shape[0])]

        return features, label
