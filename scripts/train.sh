#!/bin/bash
# TCGA-PAAD 와 CPTAC-PDA 기반 학습을 각각 별도 SLURM job으로 제출해
# train.py를 코호트별로 병렬 실행한다.
#
# 사용법:
#   bash scripts/train.sh              # tcga job + cptac job 둘 다 제출
#   bash scripts/train.sh --fusion     # 나머지 인자는 두 job에 그대로 전달됨

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$(dirname "$SCRIPT_DIR")/.logs"
mkdir -p "$LOG_DIR"

TCGA_JOB_ID=$(sbatch "${SCRIPT_DIR}/train_tcga.sh" "$@" | awk '{print $NF}')
echo "Submitted TCGA-PAAD 학습  -> job ${TCGA_JOB_ID}"

CPTAC_JOB_ID=$(sbatch "${SCRIPT_DIR}/train_cptac.sh" "$@" | awk '{print $NF}')
echo "Submitted CPTAC-PDA 학습  -> job ${CPTAC_JOB_ID}"

echo "----------------------------------------------"
echo "Monitor: squeue -j ${TCGA_JOB_ID},${CPTAC_JOB_ID}"
