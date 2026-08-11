"""
BRCA 데이터(로컬에만 있고 HPC엔 전혀 없음 — 확인됨)를 HPC로 옮기기 위한 압축 스크립트.
scripts/zip_uni2_features.py와 반대 방향(로컬 -> HPC)이지만 관례는 동일 — 프로젝트 루트
기준 상대경로 그대로 zip에 담아서, HPC에서 그 zip을 프로젝트 루트에 풀면 정확히 같은
위치에 복원된다.

담는 것: data/patches_tcga_brca/tiles/*/{coords.pt,features_uni.pt}(가장 큰 부분, 실측
~47GB), data/brca_clinical.csv, data/brca_slide_manifest.csv, data/rna_brca.csv,
data/brca_rna_gene_selection/selected_genes_top_1500.csv — scripts/train_pancancer_paad_brca.py
+ scripts/brca_common.py가 참조하는 파일 전부.

float32 텐서라 압축해도 별로 안 줄어드는 데 비해 시간만 오래 걸려서 기본은 무압축
(ZIP_STORED). 47GB라 전체를 한 번에 만드는 대신 --split-parts로 여러 zip으로 나눌 수도
있다(전송 중간에 끊겨도 처음부터 다시 안 하도록).

사용법(로컬, 프로젝트 루트에서):
    python -m scripts.zip_brca_for_hpc
    python -m scripts.zip_brca_for_hpc --out brca_for_hpc.zip

HPC에서 받은 뒤:
    cd /pub/wonseukl/Path-ViT/ && unzip -o brca_for_hpc.zip
"""
import argparse
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, default="brca_for_hpc.zip",
                         help="프로젝트 루트 기준 출력 zip 경로(기본: 루트 바로 아래)")
    parser.add_argument("--compress", action="store_true",
                         help="deflate 압축 사용(기본은 무압축 ZIP_STORED — float32 텐서라 "
                              "압축 이득이 작아 속도를 우선한다)")
    args = parser.parse_args()
    out_path = _ROOT / args.out

    files = []

    tiles_root = _ROOT / "data" / "patches_tcga_brca" / "tiles"
    slide_files = sorted(tiles_root.glob("*/coords.pt")) + sorted(tiles_root.glob("*/features_uni.pt"))
    print(f"BRCA WSI feature/coords: {len(slide_files)}개 발견 ({tiles_root})")
    files.extend(slide_files)

    small_files = [
        _ROOT / "data" / "brca_clinical.csv",
        _ROOT / "data" / "brca_slide_manifest.csv",
        _ROOT / "data" / "rna_brca.csv",
        _ROOT / "data" / "brca_rna_gene_selection" / "selected_genes_top_1500.csv",
    ]
    for f in small_files:
        if not f.exists():
            print(f"  [경고] 없음(스킵): {f}")
            continue
        files.append(f)

    if not files:
        print("담을 파일을 하나도 못 찾았습니다.")
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
    print("HPC로 전송 후: cd /pub/wonseukl/Path-ViT/ && unzip -o brca_for_hpc.zip")


if __name__ == "__main__":
    main()
