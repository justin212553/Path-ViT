"""
extract_features.py 가 생성한 features.pt 파일을 일괄 삭제하는 스크립트.

CNN 백본을 교체한 뒤 이전 캐시를 지울 때 사용한다.

사용법:
    python -m data.delete_features          # 실제 삭제
    python -m data.delete_features --dry-run  # 삭제 대상만 출력
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DataConfig
from data.patch_utils import FEATURES_FILENAME


def main():
    parser = argparse.ArgumentParser(description="features.pt 캐시 파일 삭제")
    parser.add_argument("--dry-run", action="store_true", help="삭제하지 않고 대상만 출력")
    args = parser.parse_args()

    cfg = DataConfig()
    patches_root = Path(cfg.patches_root)

    targets = sorted(patches_root.rglob(FEATURES_FILENAME))

    if not targets:
        print(f"삭제할 {FEATURES_FILENAME} 파일이 없습니다: {patches_root}")
        return

    for path in targets:
        if args.dry_run:
            print(f"[dry-run] {path}")
        else:
            path.unlink()
            print(f"삭제: {path}")

    print(f"\n총 {len(targets)}개 {'(dry-run, 실제 삭제 안 함)' if args.dry_run else '삭제 완료'}")


if __name__ == "__main__":
    main()
