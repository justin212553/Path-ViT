"""
공식 UNI2-h feature(scripts/convert_uni2h_official_features.py 산출물, features_uni2official.pt +
coords_uni2official.pt, 실측 총 ~14GB)를 HPC로 옮기기 위한 압축 스크립트. 원본 h5(TCGA-PAAD.tar.gz
+ cptac_pda.tar.gz)는 HuggingFace gated repo라 HPC에서 직접 받으려면 별도 승인/토큰 설정이
번거로우므로, 이미 로컬에서 변환까지 끝낸 최종 .pt 파일만 옮긴다(zip_brca_for_hpc.py와 동일 관례
— 프로젝트 루트 기준 상대경로 그대로 담아서 HPC에서 풀면 정확히 같은 위치에 복원됨).

float32 텐서라 압축해도 별로 안 줄어드는 데 비해 시간만 오래 걸려서 기본은 무압축(ZIP_STORED).

사용법(로컬, 프로젝트 루트에서):
    python -m scripts.zip_uni2official_features_for_hpc

HPC에서 받은 뒤:
    cd /pub/wonseukl/Path-ViT/ && unzip -o uni2official_features_for_hpc.zip
"""
import argparse
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.patch_utils import FEATURES_UNI2OFFICIAL_FILENAME, COORDS_UNI2OFFICIAL_FILENAME


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, default="uni2official_features_for_hpc.zip",
                         help="프로젝트 루트 기준 출력 zip 경로(기본: 루트 바로 아래)")
    parser.add_argument("--compress", action="store_true",
                         help="deflate 압축 사용(기본은 무압축 ZIP_STORED)")
    args = parser.parse_args()
    out_path = _ROOT / args.out

    files = []
    for tag, patches_root in [("tcga", _ROOT / "data" / "patches_tcga" / "tiles"),
                               ("cptac", _ROOT / "data" / "patches_cptac" / "tiles")]:
        feat = sorted(patches_root.glob(f"*/{FEATURES_UNI2OFFICIAL_FILENAME}"))
        coord = sorted(patches_root.glob(f"*/{COORDS_UNI2OFFICIAL_FILENAME}"))
        print(f"{tag}: feature {len(feat)}개, coords {len(coord)}개 발견 ({patches_root})")
        files.extend(feat)
        files.extend(coord)

    if not files:
        print("담을 파일을 하나도 못 찾았습니다 — scripts/convert_uni2h_official_features.py를 먼저 실행하세요.")
        return

    total_bytes = sum(f.stat().st_size for f in files)
    print(f"총 {len(files)}개 파일, {total_bytes / 1e9:.2f} GB — 압축 시작... "
          f"({'deflate' if args.compress else '무압축(ZIP_STORED)'})")

    compression = zipfile.ZIP_DEFLATED if args.compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(out_path, "w", compression=compression, allowZip64=True) as zf:
        for i, f in enumerate(files, 1):
            arcname = f.relative_to(_ROOT).as_posix()
            zf.write(f, arcname)
            if i % 200 == 0 or i == len(files):
                print(f"  {i}/{len(files)}")

    print(f"완료: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")
    print("HPC로 전송 후: cd /pub/wonseukl/Path-ViT/ && unzip -o uni2official_features_for_hpc.zip")


if __name__ == "__main__":
    main()
