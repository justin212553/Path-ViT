#!/bin/bash
# CAMELYON17 데이터 준비 파이프라인을 download → extract → eval 전처리 순서로 Slurm에 제출.
# 각 단계는 별도 sbatch job이며, --dependency=afterok로 묶어 앞 단계가 성공해야 다음 단계가 시작된다.
#
# 사용법:
#   bash scripts/data_pipeline.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/3] download 제출 (utils/dataset_download_zip.py)"
JOB_DOWNLOAD=$(sbatch --parsable "$SCRIPT_DIR/download_dataset.sh")
echo "  job id: $JOB_DOWNLOAD"

echo "[2/3] extract 제출 (utils/extract_data.py, afterok:$JOB_DOWNLOAD)"
JOB_EXTRACT=$(sbatch --parsable --dependency=afterok:$JOB_DOWNLOAD "$SCRIPT_DIR/extract_dataset.sh")
echo "  job id: $JOB_EXTRACT"

echo "[3/3] eval 전처리 제출 (data/preprocess_eval.py, afterok:$JOB_EXTRACT)"
JOB_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_EXTRACT "$SCRIPT_DIR/preprocess_eval.sh")
echo "  job id: $JOB_EVAL"

echo
echo "제출 완료: download=$JOB_DOWNLOAD → extract=$JOB_EXTRACT → eval=$JOB_EVAL"
echo "상태 확인: squeue -u \$USER"
