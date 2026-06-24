#!/bin/bash
# CAMELYON17 전체 파이프라인을 SLURM job dependency로 순차 실행.
#
# 순서: download(patient 0-49) -> extract -> preprocess
#       -> download(patient 50-99) -> extract -> preprocess
#
# 각 단계는 이전 단계가 성공(afterok)해야 시작되도록 의존성이 걸려 있어,
# 로그인 노드에서 한 번만 실행하면 큐에 6개 job이 모두 제출되고 이후는 SLURM이 순서를 보장한다.
#
# 실행:
#   bash scripts/run_pipeline_slurm.sh
set -euo pipefail

cd "$(dirname "$0")/.."

submit() {
    sbatch --parsable "$@"
}

echo "[1/6] download patient 0-49 제출"
JOB1=$(submit scripts/download_dataset.sh 0-49)
echo "  job id: $JOB1"

echo "[2/6] extract 제출 (dependency: afterok:$JOB1)"
JOB2=$(submit --dependency=afterok:$JOB1 scripts/extract_dataset.sh)
echo "  job id: $JOB2"

echo "[3/6] preprocess 제출 (dependency: afterok:$JOB2)"
JOB3=$(submit --dependency=afterok:$JOB2 scripts/preprocess.sh)
echo "  job id: $JOB3"

echo "[4/6] download patient 50-99 제출 (dependency: afterok:$JOB3)"
JOB4=$(submit --dependency=afterok:$JOB3 scripts/download_dataset.sh 50-99)
echo "  job id: $JOB4"

echo "[5/6] extract 제출 (dependency: afterok:$JOB4)"
JOB5=$(submit --dependency=afterok:$JOB4 scripts/extract_dataset.sh)
echo "  job id: $JOB5"

echo "[6/6] preprocess 제출 (dependency: afterok:$JOB5)"
JOB6=$(submit --dependency=afterok:$JOB5 scripts/preprocess.sh)
echo "  job id: $JOB6"

echo
echo "전체 체인 제출 완료: $JOB1 -> $JOB2 -> $JOB3 -> $JOB4 -> $JOB5 -> $JOB6"
echo "상태 확인: squeue -u \$USER"
