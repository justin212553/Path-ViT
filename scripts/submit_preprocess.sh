#!/bin/bash
# WSI 슬라이드를 NUM_TASKS개의 SLURM 잡으로 나눠 제출
# 사용법: bash scripts/submit_preprocess.sh [NUM_TASKS] [--workers N] [--io-threads N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
WSI_DIR="${REPO_DIR}/data/wsi_train"
LOG_DIR="${REPO_DIR}/.logs"

NUM_TASKS="${1:-5}"
shift || true          # 나머지 인자는 preprocess.py에 그대로 전달
EXTRA_ARGS="${*:---workers 10 --io-threads 4}"

mkdir -p "$LOG_DIR"

# ── WSI 슬라이드 수 확인 ───────────────────────────────────────────────────────
PATIENTS=$(ls -d "${WSI_DIR}"/patient_* 2>/dev/null | wc -l)
SLIDES=$(find "${WSI_DIR}" -name "*.tif" 2>/dev/null | wc -l)

echo "WSI root   : ${WSI_DIR}"
echo "Patients   : ${PATIENTS}"
echo "Slides     : ${SLIDES}"
echo "Tasks      : ${NUM_TASKS}  (~$((SLIDES / NUM_TASKS)) slides/task)"
echo "Extra args : ${EXTRA_ARGS:-<none>}"
echo "----------------------------------------------"

# ── SLURM 잡 제출 ─────────────────────────────────────────────────────────────
JOB_IDS=()
for i in $(seq 0 $((NUM_TASKS - 1))); do
    JOB_ID=$(sbatch \
        --job-name="preprocess_${i}" \
        --output="${LOG_DIR}/preprocess_${i}_%j.log" \
        "${SCRIPT_DIR}/preprocess.sh" \
        --task-id "$i" \
        --num-tasks "$NUM_TASKS" \
        ${EXTRA_ARGS} \
        | awk '{print $NF}')
    JOB_IDS+=("$JOB_ID")
    echo "Submitted task ${i}/${NUM_TASKS}  →  job ${JOB_ID}"
done

echo "----------------------------------------------"
echo "All job IDs: ${JOB_IDS[*]}"
echo "Monitor: squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
