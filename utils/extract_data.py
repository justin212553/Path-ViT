"""
CAMELYON17 데이터 압축 해제 스크립트

동작 순서:
  1. wsi_train/*.zip, wsi_eval/*.zip → patient_NNN/ 디렉토리로 해제

결과 구조 (patch_dataset.py 기대 형식):
  data/
    wsi_train/
      patient_000/
        r0000_c0000.png  ...
      patient_001/  ...
    wsi_eval/
      patient_004/  ...
"""
import json
import os
import shutil
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

DATA_ROOT = Path("./data")


def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def send_slack(message: str):
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return
    try:
        data = json.dumps({"text": message}).encode()
        req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[Slack] 알림 전송 실패: {e}")


def _extract_zip_flat(zip_path: Path, out_dir: Path) -> None:
    """
    zip을 out_dir 에 해제한다.
    zip 내 모든 항목이 단일 최상위 폴더 아래 있으면 그 폴더를 벗겨낸다.
    (patient_000.zip 안에 patient_000/ 폴더가 있는 경우 이중 중첩 방지)
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n]
        top_dirs = {n.split("/")[0] for n in names}

        # 단일 최상위 디렉토리이고 그 이름이 zip stem 과 같을 때 벗겨냄
        if len(top_dirs) == 1 and next(iter(top_dirs)) == zip_path.stem:
            prefix = zip_path.stem + "/"
            for info in zf.infolist():
                rel = info.filename
                if not rel.startswith(prefix):
                    continue
                rel = rel[len(prefix):]
                if not rel:
                    continue
                target = out_dir / rel
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info.filename) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(out_dir)


def extract_patient_zips(wsi_dir: Path) -> None:
    zips = sorted(wsi_dir.glob("patient_*.zip"))
    if not zips:
        print(f"  [SKIP] {wsi_dir.name}/ — patient zip 없음")
        return

    for zip_path in zips:
        out_dir = wsi_dir / zip_path.stem
        if out_dir.is_dir() and any(out_dir.iterdir()):
            print(f"  [SKIP] {zip_path.name} — 이미 해제됨")
            continue

        print(f"  [UNZIP] {zip_path.name} → {out_dir.name}/")
        out_dir.mkdir(exist_ok=True)
        _extract_zip_flat(zip_path, out_dir)

        zip_path.unlink()
        print(f"  [DEL]   {zip_path.name}")


def main() -> None:
    _load_env()
    start_time = datetime.now()

    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"data 디렉토리를 찾을 수 없음: {DATA_ROOT}")

    # ── wsi_train / wsi_eval 안의 patient zip 해제 ───────────────────────────
    print("[1/1] patient zip 해제")
    for subdir in ("wsi_train", "wsi_eval"):
        target = DATA_ROOT / subdir
        if not target.is_dir():
            print(f"  [SKIP] {subdir}/ 디렉토리 없음")
            continue
        print(f"  {subdir}/")
        extract_patient_zips(target)

    print("완료.")

    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    send_slack(
        f":white_check_mark: *CAMELYON17 압축 해제 완료*\n"
        f"> 소요 시간: {h}h {m}m {s}s"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _load_env()
        send_slack(f":x: *CAMELYON17 압축 해제 에러*\n```{type(e).__name__}: {e}```")
        raise
