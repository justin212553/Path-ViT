"""
2026-09-05: data/extract_porpoise_style_features.py가 UNI2-h 공식 추출(data/
uni2h_official_features/tcga/*.h5)의 슬라이드 목록을 기준으로 도는데, 이 목록이 PORPOISE
공식 CSV(datasets_csv/tcga_paad_all_clean.csv.zip, 377개 슬라이드 행)의 슬라이드 목록과
완전히 일치하지 않는다 — 일부 슬라이드의 .pt가 없어 DataLoader가 FileNotFoundError로 죽는다.

main.py/dataset_survival.py는 전혀 안 건드리고(원본 알고리즘 코드 보존), 실제로 존재하는
.pt 파일만 남기도록 CSV를 필터링한다 — 케이스가 슬라이드를 여러 장 가진 경우 없는 슬라이드
행만 빼고 나머지 슬라이드는 그대로 유지(케이스 자체를 통째로 빼지 않음).

원본은 .orig로 한 번만 백업해두고, 매번 그 원본에서 새로 필터링한다(재추출로 파일이 늘어나도
다시 정확하게 반영되도록).

사용법(sbatch 스크립트에서 main.py 실행 직전에 호출):
    python filter_available_slides.py --pt-files-dir <추출된 pt_files 경로>
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).parent / "datasets_csv" / "tcga_paad_all_clean.csv.zip"
ORIG_PATH = CSV_PATH.parent / "tcga_paad_all_clean.orig.csv.zip"
INNER_CSV_NAME = "tcga_paad_all_clean.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt-files-dir", type=str, required=True)
    args = parser.parse_args()
    pt_dir = Path(args.pt_files_dir)

    if not ORIG_PATH.exists():
        shutil.copy(CSV_PATH, ORIG_PATH)
        print(f"원본 백업: {ORIG_PATH}")

    df = pd.read_csv(ORIG_PATH, compression="zip")
    n_before = len(df)
    cases_before = df["case_id"].nunique()

    exists = df["slide_id"].apply(lambda sid: (pt_dir / f"{sid}.pt").exists())
    missing = df.loc[~exists, "slide_id"].tolist()
    df_filtered = df.loc[exists].reset_index(drop=True)

    # 슬라이드가 전부 없어져서 케이스 자체가 통째로 사라지는 경우도 확인
    cases_after = df_filtered["case_id"].nunique()

    print(f"pt_files_dir={pt_dir}")
    print(f"슬라이드 행: {n_before} -> {len(df_filtered)} ({len(missing)}개 제외)")
    print(f"케이스 수: {cases_before} -> {cases_after} ({cases_before - cases_after}개 케이스 완전히 사라짐)")
    if missing:
        print("제외된 슬라이드(최대 20개 표시):")
        for s in missing[:20]:
            print(f"  {s}")

    df_filtered.to_csv(CSV_PATH, index=False, compression={"method": "zip", "archive_name": INNER_CSV_NAME})
    print(f"필터링된 CSV 저장: {CSV_PATH}")


if __name__ == "__main__":
    main()
