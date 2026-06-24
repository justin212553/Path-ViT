"""
CAMELYON17 다운로드 → 압축 해제 → 전처리 파이프라인

다운로드 대상 patient ID 범위를 나눠서 여러 번 실행할 수 있도록
--patients 로 범위를 지정. 압축 해제/전처리는 매 실행마다 전체
wsi_train/ 을 대상으로 하되, 이미 처리된 파일은 각 단계에서 자동으로
skip 되므로 여러 번 실행해도 누적해서 전체가 처리된다.

실행 예시 (총 2회 실행으로 patient_000~099 전체 처리):
    # 1차: patient 1~50 다운로드
    python run_pipeline.py --patients 1-50

    # 2차: 나머지(0, 51~99) 다운로드
    python run_pipeline.py --patients 0,51-99
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAMELYON17 다운로드 → 압축 해제 → 전처리 파이프라인",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--patients", required=True,
        help='이번 실행에서 다운로드할 patient ID 범위. 예: "1-50" 또는 "0,51-99"',
    )
    parser.add_argument("--data-root", default=None, help="utils/dataset_download_zip.py 의 --data-root 로 전달")
    parser.add_argument("--workers", type=int, default=3, help="동시 다운로드 워커 수")
    parser.add_argument("--retries", type=int, default=3, help="다운로드 재시도 횟수")
    args = parser.parse_args()

    download_cmd = [
        sys.executable, "-m", "utils.dataset_download_zip",
        "--patients", args.patients,
        "--workers", str(args.workers),
        "--retries", str(args.retries),
    ]
    if args.data_root:
        download_cmd += ["--data-root", args.data_root]
    run(download_cmd)

    run([sys.executable, "-m", "utils.extract_data"])
    run([sys.executable, "-m", "data.preprocess"])

    print("\n파이프라인 완료.")


if __name__ == "__main__":
    main()
