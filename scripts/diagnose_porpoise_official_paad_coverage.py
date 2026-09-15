"""
PORPOISE 공식 mutsig 166명 코호트가 이 연구의 uni2native 타일 파이프라인 및 true-ResNet50
feature 추출 결과와 실제로 얼마나 겹치는지 진단한다.

2026-09-15: backfill(data/extract_porpoise_style_features.py --csv-path .../orig.csv.zip)을
끝까지 돌렸는데도 최종 학습 코호트가 예상(157명 근처, 166명 중 타일 자체가 없는 9명만 제외)이
아니라 137명으로 나왔다 — 그 격차가 정확히 어디서 생기는지(슬라이드 단위 vs 케이스 단위,
타일 없음 vs pt 없음 vs 케이스의 모든 슬라이드가 동시에 문제인 경우)를 명확히 짚기 위한
일회성 진단 스크립트. 코드 수정은 하지 않는다 — 순수 조회용.

사용법(HPC):
    python -m scripts.diagnose_porpoise_official_paad_coverage
"""
import zipfile
from pathlib import Path

import pandas as pd

ORIG_CSV = Path("porpoise/datasets_csv_mutsig/tcga_paad_all_clean.orig.csv.zip")
TILES_ROOT = Path("data/patches_tcga_uni2native/tiles")
PT_DIR = Path("data/porpoise_style_features/tcga/pt_files")


def slide_status(slide_id_with_svs: str) -> str:
    sid = slide_id_with_svs.rstrip(".svs")
    tile_dir = TILES_ROOT / sid
    has_tiles = tile_dir.is_dir() and any(tile_dir.glob("*.jpg"))
    has_pt = (PT_DIR / f"{sid}.pt").exists()
    if has_pt:
        return "pt_ok"
    if has_tiles:
        return "tiles_but_no_pt"
    return "no_tiles"


def main():
    with zipfile.ZipFile(ORIG_CSV) as zf:
        with zf.open(zf.namelist()[0]) as f:
            df = pd.read_csv(f)

    n_cases_total = df["case_id"].nunique()
    n_slides_total = len(df)
    print(f"공식 원본: 케이스 {n_cases_total}명, 슬라이드 {n_slides_total}개\n")

    df["status"] = df["slide_id"].apply(slide_status)

    print("=== 슬라이드 단위 상태 ===")
    print(df["status"].value_counts().to_string())
    print()

    # 케이스별로 상태 집계 -- pt_ok 슬라이드가 하나라도 있으면 그 케이스는 학습에 남는다.
    case_has_pt = df.groupby("case_id")["status"].apply(lambda s: (s == "pt_ok").any())
    n_cases_survive = int(case_has_pt.sum())
    n_cases_dropped = n_cases_total - n_cases_survive
    print(f"=== 케이스 단위 결과 ===")
    print(f"pt_ok 슬라이드가 하나라도 있는 케이스(=학습에 남는 케이스): {n_cases_survive}/{n_cases_total}")
    print(f"완전히 탈락하는 케이스: {n_cases_dropped}")
    print()

    dropped_cases = case_has_pt[~case_has_pt].index.tolist()
    print(f"=== 탈락 케이스 {len(dropped_cases)}명의 슬라이드별 상세 ===")
    detail = df[df["case_id"].isin(dropped_cases)][["case_id", "slide_id", "status"]]
    for case_id, group in detail.groupby("case_id"):
        statuses = group["status"].tolist()
        print(f"  {case_id}: 슬라이드 {len(group)}개, 상태={statuses}")

    print()
    print("=== 요약: no_tiles vs tiles_but_no_pt (탈락 케이스 기준, 케이스 내 최선 상태) ===")
    case_best_status = detail.groupby("case_id")["status"].apply(
        lambda s: "tiles_but_no_pt" if (s == "tiles_but_no_pt").any() else "no_tiles"
    )
    print(case_best_status.value_counts().to_string())


if __name__ == "__main__":
    main()
