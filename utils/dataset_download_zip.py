"""
CAMELYON17 WSI 자동 다운로드 스크립트 (zip 보존 버전)

dataset_download.py 와 동일하지만 압축 해제 없이 zip 파일 그대로 저장.

기능:
  - 최대 N개 동시 다운로드 (ThreadPoolExecutor)
  - 부분 다운로드 감지 후 재개 (Content-Length 비교)
  - 완료된 zip은 건너뜀
  - 비-TTY 환경(cluster log)에서 tqdm 자동 비활성화
  - CLI 인자로 경로·워커 수 등 모두 설정 가능
  - $DATA_ROOT 환경 변수 지원

실행 예시:
  python utils/dataset_download_zip.py --data-root /scratch/$USER/camelyon17
"""
import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ── 기본 설정 ──────────────────────────────────────────────────────────────────
BASE_URL   = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/"
    "10.5524/100001_101000/100439/CAMELYON17/training"
)
ANNO_URL   = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/"
    "10.5524/100001_101000/100439/CAMELYON17/training/lesion_annotations.zip"
)
CHUNK_SIZE = 1024 * 1024  # 1 MB

# wsi_train/wsi_eval 어디에도 아직 없는 환자만 (기존 53명은 이미 받아둔 상태라 재다운로드 제외)
MISSING_PATIENTS = [
    2, 3, 11, 18, 23, 25, 27, 28, 29,
    30, 31, 32, 33, 35, 37, 43, 47, 49,
    50, 53, 54, 55, 56, 57, 58, 59, 63,
    65, 69, 70, 71, 74, 76, 77, 78, 79,
    82, 83, 84, 85, 90, 91, 93, 94, 95,
    97, 98,
]
# ──────────────────────────────────────────────────────────────────────────────


def _setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("camelyon17_download_zip")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _url(patient_id: int) -> str:
    center = patient_id // 20
    return f"{BASE_URL}/center_{center}/patient_{patient_id:03d}.zip"


def _is_complete(zip_path: Path, expected_bytes: int) -> bool:
    if not zip_path.exists():
        return False
    if expected_bytes <= 0:
        return True
    return zip_path.stat().st_size == expected_bytes


def _download_one(
    patient_id: int,
    out_dir: Path,
    logger: logging.Logger,
    retries: int = 3,
    use_tqdm: bool = True,
) -> tuple[int, bool]:
    """단일 환자 zip 다운로드 (압축 해제 없음)."""
    zip_path = out_dir / f"patient_{patient_id:03d}.zip"
    url      = _url(patient_id)
    tag      = f"patient_{patient_id:03d}"

    for attempt in range(1, retries + 1):
        try:
            head     = requests.head(url, timeout=30, allow_redirects=True)
            expected = int(head.headers.get("content-length", 0))

            if _is_complete(zip_path, expected):
                logger.info(f"SKIP   {tag}.zip (이미 완료, {expected/1e9:.2f} GB)")
                return patient_id, True

            if zip_path.exists():
                logger.warning(f"PARTIAL  {tag}.zip 부분 파일 제거 후 재다운로드")
                zip_path.unlink()

            logger.info(f"START  {tag}.zip  ({expected/1e9:.2f} GB, 시도 {attempt}/{retries})")
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with (
                    open(zip_path, "wb") as f,
                    tqdm(
                        total=total or None,
                        unit="B", unit_scale=True, unit_divisor=1024,
                        desc=tag, leave=True,
                        disable=not use_tqdm,
                    ) as bar,
                ):
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
                        bar.update(len(chunk))

            logger.info(f"DONE   {tag}.zip → {out_dir}")
            return patient_id, True

        except Exception as exc:
            logger.warning(f"RETRY  {attempt}/{retries}  {tag}: {exc}")
            zip_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(15 * attempt)

    logger.error(f"FAIL   {tag} (모든 재시도 실패)")
    return patient_id, False


def _download_annotations(
    data_root: Path,
    logger: logging.Logger,
    use_tqdm: bool = True,
) -> bool:
    """lesion_annotations.zip 다운로드 (압축 해제 없음)."""
    zip_path = data_root / "lesion_annotations.zip"
    anno_dir = data_root / "lesion_annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)

    head     = requests.head(ANNO_URL, timeout=30, allow_redirects=True)
    expected = int(head.headers.get("content-length", 0))

    if _is_complete(zip_path, expected):
        logger.info(f"SKIP   lesion_annotations.zip (이미 완료)")
        return True

    logger.info(f"START  lesion_annotations.zip")
    try:
        with requests.get(ANNO_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with (
                open(zip_path, "wb") as f,
                tqdm(
                    total=total or None,
                    unit="B", unit_scale=True, unit_divisor=1024,
                    desc="lesion_annotations", leave=True,
                    disable=not use_tqdm,
                ) as bar,
            ):
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    bar.update(len(chunk))

        logger.info(f"DONE   lesion_annotations.zip → {zip_path}")
        return True

    except Exception as exc:
        logger.error(f"FAIL   lesion_annotations.zip: {exc}")
        zip_path.unlink(missing_ok=True)
        return False


def _run(
    tasks: list[tuple[int, Path]],
    logger: logging.Logger,
    max_workers: int = 3,
    retries: int = 3,
    use_tqdm: bool = True,
) -> tuple[list[int], list[int]]:
    done, failed = [], []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_one, pid, d, logger, retries, use_tqdm): pid
            for pid, d in tasks
        }
        for future in as_completed(futures):
            pid, ok = future.result()
            (done if ok else failed).append(pid)
            remaining = len(futures) - len(done) - len(failed)
            logger.info(f"진행  완료={len(done)}  실패={len(failed)}  남음={remaining}")

    return done, failed


def _parse_args() -> argparse.Namespace:
    default_root = os.environ.get("DATA_ROOT", "./data")

    parser = argparse.ArgumentParser(
        description="CAMELYON17 WSI 다운로드 - zip 보존 버전",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=default_root)
    parser.add_argument("--workers",   type=int, default=3)
    parser.add_argument("--retries",   type=int, default=3)
    parser.add_argument("--log-file",  default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()

    data_root = Path(args.data_root)
    log_path  = Path(args.log_file) if args.log_file else data_root / "download_zip.log"
    logger    = _setup_logging(log_path)

    use_tqdm = not args.no_progress and sys.stdout.isatty()

    wsi_dir = data_root / "wsi_train"
    wsi_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(pid, wsi_dir) for pid in MISSING_PATIENTS]

    logger.info("=" * 60)
    logger.info(f"CAMELYON17 다운로드 시작 (zip 보존)")
    logger.info(f"  data_root : {data_root.resolve()}")
    logger.info(f"  총 파일   : {len(tasks)}개  (동시 {args.workers}개)")
    logger.info(f"  wsi       → {wsi_dir}  (미보유 {len(MISSING_PATIENTS)}명만, eval/train 구분 없음)")
    logger.info("=" * 60)

    _download_annotations(data_root, logger, use_tqdm=use_tqdm)

    done, failed = _run(tasks, logger=logger,
                        max_workers=args.workers, retries=args.retries,
                        use_tqdm=use_tqdm)

    logger.info("=" * 60)
    logger.info(f"최종  완료={len(done)}  실패={len(failed)}")
    if failed:
        logger.error(f"실패 목록: {[f'patient_{p:03d}' for p in sorted(failed)]}")
    logger.info(f"로그 파일: {log_path.resolve()}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
