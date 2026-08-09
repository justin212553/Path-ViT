r"""
scripts/zip_uni2_features.py(HPC에서 실행)로 만든 uni2_features.zip을 프로젝트 루트에 그대로
풀어 data/patches_{tcga,cptac}/tiles/<slide_id>/features_uni2.pt 원래 위치에 복원한다. zip
안의 경로가 이미 프로젝트 루트 기준 상대경로(data/patches_.../...)라 압축 해제 대상 폴더만
이 프로젝트 루트로 맞추면 된다 — HPC/로컬 절대경로가 달라도 상관없다.

사용법(로컬, 프로젝트 루트에서):
    python -m scripts.unzip_uni2_features uni2_features.zip
    python -m scripts.unzip_uni2_features D:/Downloads/uni2_features.zip
"""
import argparse
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    parser = argparse.ArgumentParser(description="uni2_features.zip을 프로젝트 루트에 복원한다(로컬에서 실행)")
    parser.add_argument("zip_path", type=str, help="scripts/zip_uni2_features.py가 만든 zip 파일 경로")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"zip 파일을 찾을 수 없습니다: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # 안전 체크 — zip 안 경로가 전부 프로젝트 구조(data/patches_.../features_uni2.pt)
        # 상대경로인지 확인(절대경로/상위 디렉터리 탈출(..) 항목이 섞여 있으면 중단).
        names = zf.namelist()
        bad = [n for n in names if n.startswith("/") or ".." in Path(n).parts]
        if bad:
            raise ValueError(f"안전하지 않은 경로가 zip에 포함돼 있습니다(중단): {bad[:5]}")

        print(f"{len(names)}개 파일 압축 해제 -> {_ROOT}")
        for i, name in enumerate(names, 1):
            zf.extract(name, _ROOT)
            if i % 100 == 0 or i == len(names):
                print(f"  {i}/{len(names)}")

    print("완료 — data/patches_{tcga,cptac}/tiles/<slide_id>/features_uni2.pt 위치에 복원됨.")


if __name__ == "__main__":
    main()
