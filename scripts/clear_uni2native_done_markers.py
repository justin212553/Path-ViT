"""
extract_features_uni2native_array_hpc.sh 로그에 찍힌 "WARN 슬라이드 스킵(...)" 슬라이드들을
다시 재타일링하도록, data/preprocess.py가 보는 .done 마커를 지운다 — 마커가 없으면 다음
preprocess_uni2native_retile_array_hpc.sh 실행 때 그 슬라이드만 처음부터 다시 타일링된다
(이미 있는 다른 슬라이드는 그대로 skip).

사용법(HPC에서):
    grep -h "WARN 슬라이드 스킵" .logs/extract_uni2native_array_*.log \
        | sed -E 's/.*스킵\\(.*\\): ([^:]+):.*/\\1/' > bad_slides.txt
    python scripts/clear_uni2native_done_markers.py --dataset tcga --slide-list bad_slides.txt
    python scripts/clear_uni2native_done_markers.py --dataset cptac --slide-list bad_slides.txt

또는 슬라이드 ID를 직접 나열:
    python scripts/clear_uni2native_done_markers.py --dataset tcga --slides TCGA-XD-AAUL-01Z-00-DX1.45D24E87-C851-49EC-B106-9B3642BCF1C6
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DONE_MARKER = ".done"
TILES_ROOT = {
    "tcga":  _ROOT / "data" / "patches_tcga_uni2native" / "tiles",
    "cptac": _ROOT / "data" / "patches_cptac_uni2native" / "tiles",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["tcga", "cptac"])
    parser.add_argument("--slide-list", type=str, default=None, help="슬라이드 ID를 한 줄에 하나씩 담은 텍스트 파일")
    parser.add_argument("--slides", nargs="*", default=[], help="슬라이드 ID를 커맨드라인에 직접 나열")
    args = parser.parse_args()

    slide_ids = list(args.slides)
    if args.slide_list:
        slide_ids += [l.strip() for l in Path(args.slide_list).read_text().splitlines() if l.strip()]
    slide_ids = sorted(set(slide_ids))
    if not slide_ids:
        print("지울 슬라이드 ID가 없습니다 (--slides 또는 --slide-list 필요).")
        return

    root = TILES_ROOT[args.dataset]
    n_cleared, n_missing = 0, 0
    for slide_id in slide_ids:
        marker = root / slide_id / DONE_MARKER
        if marker.exists():
            marker.unlink()
            n_cleared += 1
            print(f"  마커 삭제: {slide_id}")
        else:
            n_missing += 1
            print(f"  [스킵] 마커 없음(이미 미완료 상태이거나 경로 오류): {slide_id}")

    print(f"\n{args.dataset}: {n_cleared}개 슬라이드 재타일링 대상으로 표시, {n_missing}개는 이미 마커 없음")
    print(f"-> sbatch sbatch/preprocess_uni2native_retile_array_hpc.sh 재제출하면 이 슬라이드들만 다시 처리됩니다.")


if __name__ == "__main__":
    main()
