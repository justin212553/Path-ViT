"""
WSI 패치 공용 유틸리티 — 패치 파일명 좌표 파싱, 정렬된 패치 목록, 표준 patch transform.

data/dataset.py(WSISurvivalDataset)와 data/extract_features.py, data/fit_clusters.py 등
패치 단위로 동작하는 모듈들이 공통으로 재사용한다.
"""
import re
from pathlib import Path

from torchvision import transforms

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
    """슬라이드 디렉터리의 패치 파일을 정렬된 순서로 나열.

    data/extract_features.py가 features.pt를 만들 때도 이 순서를 그대로 써야
    캐싱된 feature 행(row)과 패치(coords)가 어긋나지 않는다.
    """
    return sorted(list(node_dir.glob("*.png")) + list(node_dir.glob("*.jpg")))
