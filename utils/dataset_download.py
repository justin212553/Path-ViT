"""
CAMELYON17 WSI 자동 다운로드 스크립트
슈퍼컴퓨터 클러스터(Slurm) 환경 지원

기능:
  - 최대 N개 동시 다운로드 (ThreadPoolExecutor)
  - 부분 다운로드 감지 후 재개 (Content-Length 비교)
  - 완료 즉시 자동 압축 해제 → zip 삭제
  - 비-TTY 환경(cluster log)에서 tqdm 자동 비활성화
  - CLI 인자로 경로·워커 수 등 모두 설정 가능
  - $DATA_ROOT 환경 변수 지원

실행 예시:
  # 직접 실행
  python utils/dataset_download.py --data-root /scratch/$USER/camelyon17

  # Slurm
  sbatch scripts/download_camelyon17.sh
"""
import argparse
import logging
import os
import sys
import time
import zipfile
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

# eval / train 환자 목록
# annotation 있는 환자 전체 (43명, 50 slides)
# 이미 보유 중인 004, 012, 021은 디렉토리 존재로 자동 건너뜀
EVAL_PATIENTS = [
     4,  9, 10, 12, 15, 16, 17,
    20, 21, 22, 24, 34, 36, 38, 39,
    40, 41, 42, 44, 45, 46, 48, 51, 52,
    60, 61, 62, 64, 66, 67, 68, 72, 73, 75,
    80, 81, 86, 87, 88, 89, 92, 96, 99,
]
# 기존 보유 7명 + 보충 3명(006, 014, 026) = 10명
# 이미 보유 중인 000, 001, 005, 007, 008, 013, 019는 자동 건너뜀
TRAIN_PATIENTS = [0, 1, 5, 6, 7, 8, 13, 14, 19, 26]
# ──────────────────────────────────────────────────────────────────────────────


def _setup_logging(log_file: Path) -> logging.Logger:
    """파일 + 콘솔 동시 로깅 설정."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("camelyon17_download")
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
    """파일이 존재하고 서버 크기와 일치하면 True (resume skip)."""
    if not zip_path.exists():
        return False
    if expected_bytes <= 0:
        return True          # Content-Length 없으면 존재만 확인
    return zip_path.stat().st_size == expected_bytes


def _extract_zip(zip_path: Path, out_dir: Path, logger: logging.Logger,
                 keep_zip: bool = False) -> bool:
    """zip 파일을 out_dir에 압축 해제. 성공 시 zip 삭제(keep_zip=False)."""
    try:
        logger.info(f"UNZIP  {zip_path.name} → {out_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        if not keep_zip:
            zip_path.unlink()
            logger.debug(f"CLEAN  {zip_path.name} 삭제 완료")
        return True
    except zipfile.BadZipFile as exc:
        logger.error(f"BAD_ZIP  {zip_path.name}: {exc} → zip 삭제 후 재시도 필요")
        zip_path.unlink(missing_ok=True)
        return False
    except Exception as exc:
        logger.error(f"UNZIP_ERR  {zip_path.name}: {exc}")
        return False


def _download_one(
    patient_id: int,
    out_dir: Path,
    logger: logging.Logger,
    retries: int = 3,
    keep_zip: bool = False,
    use_tqdm: bool = True,
) -> tuple[int, bool]:
    """단일 환자 zip 다운로드 + 압축 해제."""
    zip_path = out_dir / f"patient_{patient_id:03d}.zip"
    url = _url(patient_id)
    tag = f"patient_{patient_id:03d}"

    # ── 이미 압축 해제된 디렉토리가 있으면 건너뜀 ──────────────────────────
    extracted_dir = out_dir / f"patient_{patient_id:03d}"
    if extracted_dir.exists():
        logger.info(f"SKIP   {tag}/ (압축 해제 완료)")
        return patient_id, True

    for attempt in range(1, retries + 1):
        try:
            # HEAD 요청으로 파일 크기 먼저 확인
            head = requests.head(url, timeout=30, allow_redirects=True)
            expected = int(head.headers.get("content-length", 0))

            if _is_complete(zip_path, expected):
                logger.info(f"SKIP   {tag}.zip (이미 완료, {expected/1e9:.2f} GB)")
            else:
                # 부분 파일이 있으면 제거
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

            # ── 압축 해제 ─────────────────────────────────────────────────
            ok = _extract_zip(zip_path, out_dir, logger, keep_zip=keep_zip)
            return patient_id, ok

        except Exception as exc:
            logger.warning(f"RETRY  {attempt}/{retries}  {tag}: {exc}")
            zip_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(15 * attempt)

    logger.error(f"FAIL   {tag} (모든 재시도 실패)")
    return patient_id, False


def _run(
    tasks: list[tuple[int, Path]],
    logger: logging.Logger,
    max_workers: int = 5,
    retries: int = 3,
    keep_zip: bool = False,
    use_tqdm: bool = True,
) -> tuple[list[int], list[int]]:
    """(patient_id, out_dir) 목록을 max_workers 병렬 처리."""
    done, failed = [], []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_one, pid, d, logger, retries, keep_zip, use_tqdm): pid
            for pid, d in tasks
        }
        for future in as_completed(futures):
            pid, ok = future.result()
            (done if ok else failed).append(pid)
            remaining = len(futures) - len(done) - len(failed)
            logger.info(
                f"진행  완료={len(done)}  실패={len(failed)}  남음={remaining}"
            )

    return done, failed


def _download_annotations(
    data_root: Path,
    logger: logging.Logger,
    keep_zip: bool = False,
    use_tqdm: bool = True,
) -> bool:
    """lesion_annotations.zip 다운로드 후 data/lesion_annotations/ 에 압축 해제."""
    anno_dir = data_root / "lesion_annotations"
    zip_path = data_root / "lesion_annotations.zip"

    if anno_dir.exists() and any(anno_dir.glob("*.xml")):
        logger.info(f"SKIP   lesion_annotations/ (이미 존재: {len(list(anno_dir.glob('*.xml')))}개 XML)")
        return True

    anno_dir.mkdir(parents=True, exist_ok=True)

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

        logger.info(f"UNZIP  lesion_annotations.zip → {anno_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # zip 내부 구조에 관계없이 XML 파일만 anno_dir 바로 아래로 추출
            for member in zf.namelist():
                if member.endswith(".xml"):
                    zf.extract(member, data_root)
                    extracted = data_root / member
                    target    = anno_dir / extracted.name
                    if extracted != target:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        extracted.replace(target)

        if not keep_zip:
            zip_path.unlink()

        xml_count = len(list(anno_dir.glob("*.xml")))
        logger.info(f"DONE   lesion_annotations/ ({xml_count}개 XML)")
        return True

    except Exception as exc:
        logger.error(f"FAIL   lesion_annotations.zip: {exc}")
        zip_path.unlink(missing_ok=True)
        return False


def _parse_args() -> argparse.Namespace:
    # $DATA_ROOT 환경 변수 우선, 없으면 ../data
    default_root = os.environ.get("DATA_ROOT", "../data")

    parser = argparse.ArgumentParser(
        description="CAMELYON17 WSI 다운로드 (클러스터 지원)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", default=default_root,
        help="데이터 저장 루트 경로 (환경 변수 $DATA_ROOT 우선)",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="동시 다운로드 스레드 수",
    )
    parser.add_argument(
        "--retries", type=int, default=3,
        help="실패 시 재시도 횟수",
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="압축 해제 후 zip 파일 보존 (기본: 삭제)",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="eval 세트만 다운로드",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="train 세트만 다운로드",
    )
    parser.add_argument(
        "--log-file", default=None,
        help="로그 파일 경로 (기본: <data-root>/download.log)",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="tqdm 진행 바 비활성화 (비-TTY 환경에서는 자동 비활성화)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    data_root = Path(args.data_root)
    log_path  = Path(args.log_file) if args.log_file else data_root / "download.log"
    logger    = _setup_logging(log_path)

    # TTY 여부로 tqdm 자동 결정 (cluster job 로그에 제어문자 남지 않도록)
    use_tqdm = not args.no_progress and sys.stdout.isatty()

    eval_dir  = data_root / "wsi_eval"
    train_dir = data_root / "wsi_train"
    eval_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    # 다운로드 대상 구성
    tasks = []
    if not args.train_only:
        tasks += [(pid, eval_dir)  for pid in EVAL_PATIENTS]
    if not args.eval_only:
        tasks += [(pid, train_dir) for pid in TRAIN_PATIENTS]

    logger.info("=" * 60)
    logger.info(f"CAMELYON17 다운로드 시작")
    logger.info(f"  data_root : {data_root.resolve()}")
    logger.info(f"  총 파일   : {len(tasks)}개  (동시 {args.workers}개)")
    logger.info(f"  eval      → {eval_dir}  ({len(EVAL_PATIENTS)}명)")
    logger.info(f"  train     → {train_dir} ({len(TRAIN_PATIENTS)}명)")
    logger.info(f"  keep_zip  : {args.keep_zip}")
    logger.info(f"  tqdm      : {use_tqdm}")
    logger.info("=" * 60)

    # lesion_annotations 먼저 다운로드
    _download_annotations(data_root, logger, keep_zip=args.keep_zip, use_tqdm=use_tqdm)

    done, failed = _run(
        tasks,
        logger=logger,
        max_workers=args.workers,
        retries=args.retries,
        keep_zip=args.keep_zip,
        use_tqdm=use_tqdm,
    )

    logger.info("=" * 60)
    logger.info(f"최종  완료={len(done)}  실패={len(failed)}")
    if failed:
        logger.error(f"실패 목록: {[f'patient_{p:03d}' for p in sorted(failed)]}")
    logger.info(f"로그 파일: {log_path.resolve()}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
