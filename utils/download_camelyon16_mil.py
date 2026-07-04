"""
torchmil/Camelyon16_MIL(Hugging Face) 사전 추출 feature 다운로드 스크립트

WSI 다운로드 → data/preprocess.py 타일링 → data/extract_features.py CNN feature 추출
전체 파이프라인을 건너뛰고, torchmil이 이미 만들어둔 CAMELYON16 패치 feature를 받아
data/camelyon16_dataset.py가 기대하는 형식으로 변환한다.

기본값은 features="resnet50_bt" (ResNet50 + Barlow Twins, Kang et al. 2023 / Lunit이
TCGA 1900만 패치로 사전학습) — 지금 프로젝트가 쓰는 CNN backbone(1aurent/resnet50.lunit_swav,
같은 Kang et al. 2023 벤치마크의 SwAV 버전)과 같은 계열이라 proj 레이어 입력 차원만 맞으면
그대로 이어붙일 수 있다.

원본: https://huggingface.co/datasets/torchmil/Camelyon16_MIL
  dataset/patches_512/features/features_{resnet50_bt,resnet50,UNI}.tar.gz
  dataset/patches_512/coords.tar.gz
  dataset/patches_512/labels.tar.gz
  dataset/patches_512/patch_labels.tar.gz

주의: 내부 tar.gz의 정확한 하위 폴더 구조는 다운로드 전에 확인하지 않았다. 압축 해제 후
슬라이드 ID(파일명 stem)를 기준으로 재귀적으로 매칭하므로 폴더 깊이가 달라도 동작하지만,
coords/labels/features 세 종류의 파일명이 서로 다르면 매칭에 실패할 수 있다 — 그 경우
--keep-raw로 재실행해 data/_torchmil_staging 아래 실제 구조를 확인할 것.

출력 (camelyon16_dataset.py 기대 형식):
    <data_root>/<slide_id>/features.pt   (N_patches, feature_dim) float32
    <data_root>/<slide_id>/coords.pt     (N_patches, 2) int64  [row, col]
    <data_root>/reference.csv            공식 reference.csv 형식 (image_name,label,Type)
                                          → train.py --reference-csv로 넘기면 test_* 라벨도 사용 가능

실행 예시:
    python -m utils.download_camelyon16_mil --data-root data/patches
"""
import argparse
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import load_env, send_slack

REPO_ID = "torchmil/Camelyon16_MIL"
PATCH_SIZE = 512  # torchmil 레포에 512만 존재

METADATA_ARCHIVES = {
    "coords":       f"dataset/patches_{PATCH_SIZE}/coords.tar.gz",
    "labels":       f"dataset/patches_{PATCH_SIZE}/labels.tar.gz",
    "patch_labels": f"dataset/patches_{PATCH_SIZE}/patch_labels.tar.gz",
}
FEATURES_ARCHIVE_TMPL = f"dataset/patches_{PATCH_SIZE}/features/features_{{name}}.tar.gz"

FEATURES_FILENAME = "features.pt"
COORDS_FILENAME   = "coords.pt"


def _download_and_extract(repo_file: str, staging_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    print(f"  다운로드: {repo_file}")
    tar_path = Path(hf_hub_download(
        REPO_ID, filename=repo_file, repo_type="dataset", local_dir=str(staging_dir),
    ))
    out_dir = tar_path.parent / tar_path.name.replace(".tar.gz", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  압축 해제 → {out_dir}")
    with tarfile.open(tar_path) as tf:
        try:
            tf.extractall(out_dir, filter="data")
        except TypeError:
            tf.extractall(out_dir)  # Python < 3.12: filter 인자 없음

    tar_path.unlink()  # 압축 해제 직후 삭제 (features 아카이브가 ~26GB라 디스크 절약 필수)
    return out_dir


def _index_npy(root: Path) -> dict[str, Path]:
    """root 아래 재귀적으로 *.npy를 찾아 {slide_id: path} 매핑 (내부 폴더 구조 불확실성 대비)."""
    return {p.stem: p for p in root.rglob("*.npy")}


def _load_scalar_label(path: Path) -> int:
    return int(np.load(path).reshape(-1)[0])


def _load_coords(path: Path) -> torch.Tensor:
    coords = torch.from_numpy(np.load(path)).long()
    if coords.ndim == 2 and coords.shape[1] != 2 and coords.shape[0] == 2:
        coords = coords.T
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise RuntimeError(f"{path}: coords shape이 (N, 2)가 아님 — {tuple(coords.shape)}")
    return coords


def main():
    load_env()
    start_time = datetime.now()
    args = _parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    staging = Path(args.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    coords_dir   = _download_and_extract(METADATA_ARCHIVES["coords"], staging)
    labels_dir   = _download_and_extract(METADATA_ARCHIVES["labels"], staging)
    features_dir = _download_and_extract(
        FEATURES_ARCHIVE_TMPL.format(name=args.features), staging
    )

    coords_idx   = _index_npy(coords_dir)
    labels_idx   = _index_npy(labels_dir)
    features_idx = _index_npy(features_dir)

    slide_ids = sorted(set(coords_idx) & set(labels_idx) & set(features_idx))
    if not slide_ids:
        raise RuntimeError(
            "coords/labels/features 파일명이 서로 매칭되지 않습니다 — torchmil 데이터셋 "
            f"내부 구조 확인 필요 (coords={len(coords_idx)}개, labels={len(labels_idx)}개, "
            f"features={len(features_idx)}개). --keep-raw로 재실행 후 {staging} 확인."
        )
    missing = (set(coords_idx) | set(labels_idx) | set(features_idx)) - set(slide_ids)
    if missing:
        print(f"  경고: 세 종류 파일이 모두 있지 않아 제외됨 ({len(missing)}개): {sorted(missing)[:5]}...")

    print(f"변환 대상: {len(slide_ids)}개 슬라이드")
    ref_rows = []
    for i, slide_id in enumerate(slide_ids, 1):
        out_dir = data_root / slide_id
        out_dir.mkdir(parents=True, exist_ok=True)

        label = _load_scalar_label(labels_idx[slide_id])
        ref_rows.append((slide_id, "Tumor" if label == 1 else "Normal"))

        if (out_dir / FEATURES_FILENAME).exists() and (out_dir / COORDS_FILENAME).exists():
            continue  # 중간에 끊긴 잡을 재실행할 때 이미 변환된 슬라이드는 건너뜀

        features = torch.from_numpy(np.load(features_idx[slide_id])).float()
        coords   = _load_coords(coords_idx[slide_id])
        if len(features) != len(coords):
            raise RuntimeError(
                f"{slide_id}: features({len(features)})/coords({len(coords)}) 패치 수 불일치"
            )

        torch.save(features, out_dir / FEATURES_FILENAME)
        torch.save(coords, out_dir / COORDS_FILENAME)

        if i % 50 == 0 or i == len(slide_ids):
            print(f"  [{i}/{len(slide_ids)}] {slide_id}  (features={tuple(features.shape)})")

    ref_csv = data_root / "reference.csv"
    with open(ref_csv, "w", encoding="utf-8") as f:
        for slide_id, label_str in ref_rows:
            f.write(f"{slide_id},{label_str},\n")
    print(f"reference.csv 작성 완료 → {ref_csv} ({len(ref_rows)}줄)")

    if not args.keep_raw:
        print(f"임시 압축 해제 폴더 정리: {staging}")
        shutil.rmtree(staging, ignore_errors=True)

    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    print(f"완료: {len(slide_ids)}개 슬라이드 → {data_root}")
    send_slack(
        f":white_check_mark: *CAMELYON16 torchmil({args.features}) feature 다운로드 완료*\n"
        f"> 슬라이드: {len(slide_ids)}개 → `{data_root}`\n"
        f"> 소요 시간: {h}h {m}m {s}s"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="torchmil/Camelyon16_MIL feature 다운로드 및 camelyon16_dataset.py 형식 변환",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="data/patches",
                         help="config.py DataConfig.patches_root와 동일해야 함")
    parser.add_argument("--features", default="resnet50_bt",
                         choices=["resnet50_bt", "resnet50", "UNI"])
    parser.add_argument("--staging-dir", default="data/_torchmil_staging",
                         help="tar.gz 다운로드/압축 해제용 임시 폴더")
    parser.add_argument("--keep-raw", action="store_true",
                         help="변환 후 임시 압축 해제 폴더(원본 npy)를 지우지 않음 (디버깅용)")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        load_env()
        send_slack(f":x: *CAMELYON16 torchmil 다운로드 에러*\n```{type(e).__name__}: {e}```")
        raise
