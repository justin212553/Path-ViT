"""
`scripts/_download_brca_hf.py`로 받은 TCGA-BRCA raw HF 파일(UNI backbone feature .pt +
패치 좌표 .h5, Dearcat/CPathPatchFeature)을 `data/brca_clinical.csv`(scripts/extract_brca_labels.py
산출물)와 case_id 기준으로 join하고, 슬라이드 단위로 정리한다.

원본 파일명은 "<case_barcode>-<sample>-<portion>-<plate>-<seq><letter>.<uuid>.pt" 형식이라
(예: TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-....pt), case_id는 앞 3개 바코드 세그먼트(TCGA-XX-XXXX)다.

우리 기존 파이프라인(data/dataset.py::WSISurvivalDataset)은 패치 "이미지 파일"이 있고 좌표를
파일명에서 파싱하는 구조라(list_patch_paths), 이 데이터(이미지 없이 이미 pooled된 feature +
별도 좌표 배열)와 형식이 달라 바로 재사용은 안 된다 — 대신 슬라이드당 하나의 .pt 파일로
{"features", "coords", "case_id", "OS_time", "OS_event", "age_years", "sex"}를 묶어 저장한다.
학습 스크립트(추후 작성)는 이 슬라이드 단위 .pt만 읽으면 된다.

출력:
    data/patches_tcga_brca/tiles/<slide_barcode>/features_uni.pt   (N_patches, 1024) float32
    data/patches_tcga_brca/tiles/<slide_barcode>/coords.pt         (N_patches, 2) int64
    data/brca_slide_manifest.csv   slide_id, case_id, n_patches, OS_time, OS_event, age_years, sex

사용법:
    python -m scripts.prepare_brca_data
"""
from pathlib import Path

import h5py
import pandas as pd
import torch
from tqdm import tqdm

RAW_ROOT = Path("data/raw_brca_hf/brca")
UNI_DIR = RAW_ROOT / "uni" / "pt_files"
COORDS_DIR = RAW_ROOT / "patches"
CLINICAL_PATH = Path("data/brca_clinical.csv")
OUT_ROOT = Path("data/patches_tcga_brca/tiles")
MANIFEST_PATH = Path("data/brca_slide_manifest.csv")


def _case_id_from_slide_filename(stem: str) -> str:
    """"TCGA-3C-AALI-01Z-00-DX1.<uuid>" -> "TCGA-3C-AALI" (앞 3개 바코드 세그먼트)."""
    barcode = stem.split(".")[0]
    parts = barcode.split("-")
    return "-".join(parts[:3])


def main():
    clinical = pd.read_csv(CLINICAL_PATH).set_index("case_id")
    print(f"임상 라벨 case 수: {len(clinical)}")

    uni_files = sorted(UNI_DIR.glob("*.pt"))
    print(f"HF 다운로드된 슬라이드(uni) 수: {len(uni_files)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    skipped_no_clinical = 0
    skipped_no_coords = 0

    for pt_path in tqdm(uni_files, desc="슬라이드 정리"):
        slide_stem = pt_path.stem  # "TCGA-3C-AALI-01Z-00-DX1.<uuid>"
        case_id = _case_id_from_slide_filename(slide_stem)

        if case_id not in clinical.index:
            skipped_no_clinical += 1
            continue

        coords_path = COORDS_DIR / f"{slide_stem}.h5"
        if not coords_path.exists():
            skipped_no_coords += 1
            continue

        features = torch.load(pt_path, weights_only=True)
        with h5py.File(coords_path, "r") as f:
            coords = torch.from_numpy(f["coords"][:]).long()

        assert features.shape[0] == coords.shape[0], (
            f"{slide_stem}: feature 행 수({features.shape[0]})와 좌표 수({coords.shape[0]}) 불일치"
        )

        slide_dir = OUT_ROOT / slide_stem
        slide_dir.mkdir(parents=True, exist_ok=True)
        torch.save(features, slide_dir / "features_uni.pt")
        torch.save(coords, slide_dir / "coords.pt")

        row = clinical.loc[case_id]
        manifest_rows.append({
            "slide_id": slide_stem,
            "case_id": case_id,
            "n_patches": int(features.shape[0]),
            "OS_time": float(row["OS_time"]),
            "OS_event": int(row["OS_event"]),
            "age_years": float(row["age_years"]),
            "sex": row["sex"],
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(MANIFEST_PATH, index=False)

    print(f"\n임상 라벨 없어서 제외: {skipped_no_clinical}")
    print(f"좌표 파일 없어서 제외: {skipped_no_coords}")
    print(f"최종 슬라이드 수: {len(manifest)}  (case 수: {manifest['case_id'].nunique()})")
    print(f"event(Dead)={manifest['OS_event'].sum()}  censored(Alive)={(manifest['OS_event']==0).sum()}"
          if len(manifest) else "")
    print(f"슬라이드당 패치 수: 평균 {manifest['n_patches'].mean():.0f}  중앙값 {manifest['n_patches'].median():.0f}"
          if len(manifest) else "")
    print(f"매니페스트 저장: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
