#!/bin/bash
# TCGA-PAAD 와 CPTAC-PDA를 각각 별도 SLURM job으로 제출해 data/preprocess.py의
# tile 추출 + CNN feature 추출(utils/extract_features.py)을 코호트별로 병렬 실행한다.
#
# 사용법:
#   bash scripts/preprocess.sh                # tcga job + cptac job 둘 다 제출
#   bash scripts/preprocess.sh --tiles-only    # 나머지 인자는 두 job에 그대로 전달됨

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$(dirname "$SCRIPT_DIR")/.logs"
mkdir -p "$LOG_DIR"

TCGA_JOB_ID=$(sbatch "${SCRIPT_DIR}/preprocess_tcga.sh" "$@" | awk '{print $NF}')
echo "Submitted TCGA-PAAD preprocess  -> job ${TCGA_JOB_ID}"

CPTAC_JOB_ID=$(sbatch "${SCRIPT_DIR}/preprocess_cptac.sh" "$@" | awk '{print $NF}')
echo "Submitted CPTAC-PDA preprocess  -> job ${CPTAC_JOB_ID}"

echo "----------------------------------------------"
echo "Monitor: squeue -j ${TCGA_JOB_ID},${CPTAC_JOB_ID}"
