"""
data/patches_{tcga,cptac}_uni2native/tiles/*/features_uni2.pt 가 온전한지 검증하고, 깨진(잘린)
파일은 지운다.

utils/extract_features.py의 skip 로직(2026-08-13 이전)이 out_path.exists()만 보고 판단해서,
torch.save() 도중 SLURM TIME LIMIT 등으로 job이 죽으면 잘린 파일이 "이미 완료"로 오판돼 영원히
재생성되지 않을 위험이 있었다(원자적 rename으로 이후 쓰기는 고쳤지만 — 이미 그 버그 상태에서
쓰였을 수 있는 기존 파일은 이 스크립트로 따로 검증해야 함).

검증 기준: torch.load()가 성공하고, 2차원 float 텐서이며 shape[0]>0, shape[1]==1536(UNI2-h
embed dim)인지 확인. 하나라도 어긋나면 삭제.

사용법: python scripts/verify_uni2native_features.py [--delete]
    --delete 없이 돌리면 문제 있는 파일 목록만 보여주고 실제로 지우지는 않는다(기본은 dry-run).
"""
import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.patch_utils import FEATURES_UNI2_FILENAME

ROOTS = {
    "tcga":  _ROOT / "data" / "patches_tcga_uni2native" / "tiles",
    "cptac": _ROOT / "data" / "patches_cptac_uni2native" / "tiles",
}
EXPECTED_DIM = 1536  # UNI2-h


def _is_valid(path: Path) -> tuple[bool, str]:
    try:
        t = torch.load(path, weights_only=True)
    except Exception as e:
        return False, f"load 실패: {type(e).__name__}: {e}"
    if not torch.is_tensor(t):
        return False, f"텐서가 아님: {type(t)}"
    if t.ndim != 2 or t.shape[0] == 0 or t.shape[1] != EXPECTED_DIM:
        return False, f"shape 이상: {tuple(t.shape)}"
    if not torch.isfinite(t).all():
        return False, "NaN/Inf 포함"
    return True, ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delete", action="store_true", help="문제 있는 파일을 실제로 삭제한다(기본: dry-run)")
    args = parser.parse_args()

    for tag, root in ROOTS.items():
        if not root.exists():
            print(f"{tag}: {root} 없음 — 스킵")
            continue
        feat_paths = sorted(root.glob(f"*/{FEATURES_UNI2_FILENAME}"))
        n_ok, bad = 0, []
        for p in tqdm(feat_paths, desc=tag):
            ok, reason = _is_valid(p)
            if ok:
                n_ok += 1
            else:
                bad.append((p, reason))

        print(f"{tag}: 총 {len(feat_paths)}개 중 정상 {n_ok}개, 문제 {len(bad)}개")
        for p, reason in bad:
            print(f"  [문제] {p.parent.name}: {reason}")
            if args.delete:
                p.unlink()
        if bad and not args.delete:
            print(f"  -> --delete 없이 실행돼 실제로 지우지 않았습니다. 지우려면 --delete를 붙여 다시 실행하세요.")


if __name__ == "__main__":
    main()
