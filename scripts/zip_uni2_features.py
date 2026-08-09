"""
UNI2-h(ViT-H/14, models/uni2_encoder.py) feature 사전추출 산출물(features_uni2.pt)을 HPC에서
로컬로 옮기기 위한 압축 스크립트. data/patches_{tcga,cptac}/tiles/<slide_id>/features_uni2.pt
전부를 프로젝트 루트 기준 상대경로 그대로 zip에 담는다 — 로컬에서
scripts/unzip_uni2_features.py로 그 zip을 프로젝트 루트에 풀면 정확히 같은 위치
(data/patches_{tcga,cptac}/tiles/<slide_id>/features_uni2.pt)에 복원된다.

float32 텐서라 압축해도 별로 안 줄어드는 데 비해 시간만 오래 걸려서 기본은 무압축
(ZIP_STORED)이다 — 전송 속도가 병목이면 --compress로 deflate 압축을 켤 수 있다.

사용법(HPC, 프로젝트 루트에서):
    python -m scripts.zip_uni2_features
    python -m scripts.zip_uni2_features --out uni2_features.zip --datasets tcga cptac
    python -m scripts.zip_uni2_features --compress   # 용량이 더 중요하면
"""
import argparse
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DataConfig
from data.dataset import PATCHES_ROOT_ATTRS


def main():
    parser = argparse.ArgumentParser(description="features_uni2.pt 전부를 zip으로 묶는다(HPC에서 실행)")
    parser.add_argument("--datasets", nargs="+", default=["tcga", "cptac"], choices=["tcga", "cptac"])
    parser.add_argument("--out", type=str, default="uni2_features.zip",
                         help="프로젝트 루트 기준 출력 zip 경로(기본: 루트 바로 아래)")
    parser.add_argument("--compress", action="store_true",
                         help="deflate 압축 사용(기본은 무압축 ZIP_STORED — float32라 압축 이득이 "
                              "작아 속도를 우선한다)")
    args = parser.parse_args()

    cfg = DataConfig()
    out_path = _ROOT / args.out

    files = []
    for ds in args.datasets:
        patches_root = _ROOT / getattr(cfg, PATCHES_ROOT_ATTRS[ds]) / "tiles"
        found = sorted(patches_root.glob("*/features_uni2.pt"))
        print(f"{ds}: {len(found)}개 발견 ({patches_root})")
        files.extend(found)

    if not files:
        print("features_uni2.pt를 하나도 못 찾았습니다 — 추출이 끝났는지 먼저 확인하세요.")
        return

    total_bytes = sum(f.stat().st_size for f in files)
    print(f"총 {len(files)}개 파일, {total_bytes / 1e9:.2f} GB — 압축 시작... "
          f"({'deflate' if args.compress else '무압축(ZIP_STORED)'})")

    compression = zipfile.ZIP_DEFLATED if args.compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(out_path, "w", compression=compression) as zf:
        for i, f in enumerate(files, 1):
            arcname = f.relative_to(_ROOT).as_posix()  # 예: data/patches_tcga/tiles/<slide_id>/features_uni2.pt
            zf.write(f, arcname)
            if i % 100 == 0 or i == len(files):
                print(f"  {i}/{len(files)}")

    print(f"완료: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")
    print("로컬로 옮긴 뒤: python -m scripts.unzip_uni2_features <받은 zip 경로>")


if __name__ == "__main__":
    main()
