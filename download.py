"""
CAMELYON17 WSI 자동 다운로드 스크립트

최대 5개 동시 다운로드, 완료 즉시 다음 대기열 처리.
실행:
    python download.py

로그:
    download.log  (타임스탬프 포함, 실패·성공 기록)
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ── 설정 ──────────────────────────────────────────────────────────────────────
# GigaDB URL 구조 — 실제 경로가 다르면 이 템플릿만 수정
# 예시: http://gigadb.org/pub/10.5524/100001_101000/100439/centre_0/patient_000.zip
BASE_URL    = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100439/CAMELYON17/training"
DATA_ROOT   = Path("../data")
MAX_WORKERS = 5
RETRIES     = 3
CHUNK_SIZE  = 1024 * 1024  # 1 MB
# ─────────────────────────────────────────────────────────────────────────────

# eval용: annotation 있으나 WSI 없는 환자 (40명)
EVAL_PATIENTS = [
     9, 10, 15, 16, 17,
    20, 22, 24, 34, 36, 38, 39,
    40, 41, 42, 44, 45, 46, 48, 51, 52,
    60, 61, 62, 64, 66, 67, 68, 72, 73, 75,
    80, 81, 86, 87, 88, 89, 92, 96, 99,
]

# train 보충용: eval로 이동한 patient_004/012/021 대체
TRAIN_PATIENTS = [6, 14, 26]

# ── 로그 설정 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("download.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def _url(patient_id: int) -> str:
    center = patient_id // 20
    return f"{BASE_URL}/center_{center}/patient_{patient_id:03d}.zip"


def _download_one(patient_id: int, out_dir: Path) -> tuple[int, bool]:
    zip_path = out_dir / f"patient_{patient_id:03d}.zip"

    if zip_path.exists():
        log.info(f"SKIP   patient_{patient_id:03d}.zip (already exists)")
        return patient_id, True

    url = _url(patient_id)
    for attempt in range(1, RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with (
                    open(zip_path, "wb") as f,
                    tqdm(
                        total=total, unit="B", unit_scale=True, unit_divisor=1024,
                        desc=f"patient_{patient_id:03d}", leave=True,
                    ) as bar,
                ):
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
                        bar.update(len(chunk))

            log.info(f"DONE   patient_{patient_id:03d}.zip → {out_dir}")
            return patient_id, True

        except Exception as exc:
            log.warning(f"RETRY  {attempt}/{RETRIES}  patient_{patient_id:03d}: {exc}")
            if zip_path.exists():
                zip_path.unlink()       # 부분 파일 제거
            if attempt < RETRIES:
                time.sleep(10 * attempt)

    log.error(f"FAIL   patient_{patient_id:03d}.zip (모든 재시도 실패)")
    return patient_id, False


def _run(tasks: list[tuple[int, Path]]):
    """(patient_id, out_dir) 목록을 MAX_WORKERS 병렬로 처리."""
    done, failed = [], []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_download_one, pid, d): pid for pid, d in tasks}
        for future in as_completed(futures):
            pid, ok = future.result()
            (done if ok else failed).append(pid)
            remaining = len(futures) - len(done) - len(failed)
            log.info(f"진행  완료={len(done)}  실패={len(failed)}  남음={remaining}")

    return done, failed


def main():
    eval_dir  = DATA_ROOT / "wsi_eval"
    train_dir = DATA_ROOT / "wsi_train"
    eval_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    tasks = (
        [(pid, eval_dir)  for pid in EVAL_PATIENTS] +
        [(pid, train_dir) for pid in TRAIN_PATIENTS]
    )

    log.info(f"다운로드 시작: 총 {len(tasks)}개 파일 (동시 {MAX_WORKERS}개)")
    log.info(f"  eval  → {eval_dir}  ({len(EVAL_PATIENTS)}명)")
    log.info(f"  train → {train_dir} ({len(TRAIN_PATIENTS)}명)")

    done, failed = _run(tasks)

    log.info("=" * 50)
    log.info(f"완료: {len(done)}개 / 실패: {len(failed)}개")
    if failed:
        log.error(f"실패 목록: {[f'patient_{p:03d}' for p in sorted(failed)]}")


if __name__ == "__main__":
    main()
