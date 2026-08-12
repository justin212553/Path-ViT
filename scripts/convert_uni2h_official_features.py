"""
MahmoodLab 공식 UNI2-h feature(256px@20x, data/uni2h_official_features/{tcga,cptac}/*.h5)를
우리 학습 파이프라인이 읽을 수 있는 형태(슬라이드 디렉토리 아래 features_uni2official.pt +
coords_uni2official.pt)로 변환한다.

h5 파일명(=attrs['name'])이 우리 자체 patches 디렉토리명(data/patches_{tcga,cptac}/tiles/<slide_id>/)과
정확히 일치함을 전제한다(TCGA-PAAD/CPTAC-PDA 둘 다 확인됨, 2026-08-12). patch grid 자체가
우리 자체 추출본과 전혀 다르므로(개수/좌표 불일치) coords도 h5의 coords_patching을 그대로 쓰고,
data/dataset.py::_load_slide는 이 두 파일을 함께 읽어 list_patch_paths/파일명-파싱 coords 경로를
타지 않는다(--backbone uni2official).

사용법: python scripts/convert_uni2h_official_features.py
"""
import sys
from pathlib import Path

import h5py
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.patch_utils import FEATURES_UNI2OFFICIAL_FILENAME, COORDS_UNI2OFFICIAL_FILENAME

SRC_DIRS = {
    "tcga":  _ROOT / "data" / "uni2h_official_features" / "tcga",
    "cptac": _ROOT / "data" / "uni2h_official_features" / "cptac",
}
DST_ROOTS = {
    "tcga":  _ROOT / "data" / "patches_tcga" / "tiles",
    "cptac": _ROOT / "data" / "patches_cptac" / "tiles",
}


def main():
    for tag, src_dir in SRC_DIRS.items():
        h5_files = sorted(src_dir.glob("*.h5"))
        dst_root = DST_ROOTS[tag]
        n_ok, n_skip = 0, 0
        for h5_path in tqdm(h5_files, desc=tag):
            slide_id = h5_path.stem
            dst_dir = dst_root / slide_id
            if not dst_dir.exists():
                n_skip += 1
                continue
            with h5py.File(h5_path, "r") as f:
                features = torch.from_numpy(f["features"][0]).float()   # (N, 1536)
                coords = torch.from_numpy(f["coords_patching"][()]).long()  # (N, 2)
            coords = coords.clone()
            coords[:, 0] -= coords[:, 0].min()
            coords[:, 1] -= coords[:, 1].min()
            torch.save(features, dst_dir / FEATURES_UNI2OFFICIAL_FILENAME)
            torch.save(coords, dst_dir / COORDS_UNI2OFFICIAL_FILENAME)
            n_ok += 1
        print(f"{tag}: 변환 {n_ok}개, 매칭 안 되는 슬라이드(우리 patches 디렉토리 없음) {n_skip}개")


if __name__ == "__main__":
    main()
