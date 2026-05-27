#!/bin/bash
# ============================================================
# CAMELYON17 다운로드 Slurm 배치 스크립트
# 사용법: sbatch scripts/download_camelyon17.sh
# ============================================================

#SBATCH --job-name=camelyon17_download
#SBATCH --output=logs/download_%j.out      # %j = job ID
#SBATCH --error=logs/download_%j.err
#SBATCH --time=12:00:00                    # 최대 12시간 (WSI 파일 크기에 따라 조정)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                  # 다운로드 스레드 수와 맞춤
#SBATCH --mem=16G
# --- 아래 두 줄은 클러스터에 맞게 수정 ---
##SBATCH --partition=your_partition        # 파티션명 (주석 해제 후 수정)
##SBATCH --account=your_account           # 계정명 (주석 해제 후 수정)

# ── 환경 설정 ────────────────────────────────────────────────────────────────
set -euo pipefail

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Start      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 로그 디렉토리 생성
mkdir -p logs

# ── 데이터 저장 경로 설정 ─────────────────────────────────────────────────────
# 클러스터의 scratch 디렉토리를 사용하는 것을 권장
# $SCRATCH, $WORK, $PROJECT 등은 클러스터마다 다름 — 실제 경로로 수정하세요
export DATA_ROOT="${SCRATCH:-$HOME}/camelyon17"
echo "DATA_ROOT  : $DATA_ROOT"

# ── Python 환경 활성화 ────────────────────────────────────────────────────────
# 방법 1: conda 사용 (클러스터에 conda/mamba가 있는 경우)
# module load anaconda3          # 또는 miniforge, mambaforge 등
# conda activate path_vit        # 환경 이름 수정

# 방법 2: venv 사용
# source $HOME/envs/path_vit/bin/activate

# 방법 3: module + pip 가상환경
# module load python/3.10
# source $HOME/.venvs/path_vit/bin/activate

# ↑ 위 세 가지 중 하나를 주석 해제하고 환경명을 맞게 수정하세요

# 필수 패키지 확인
python -c "import requests, tqdm" 2>/dev/null || {
    echo "[ERROR] requests 또는 tqdm 패키지가 없습니다."
    echo "        pip install requests tqdm 을 실행하거나 환경을 활성화하세요."
    exit 1
}

# ── 프로젝트 루트로 이동 ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "Project    : $PROJECT_ROOT"

# ── 다운로드 실행 ─────────────────────────────────────────────────────────────
# --workers 8 : #SBATCH --cpus-per-task=8 과 맞춤
# --no-progress: Slurm 로그 파일에 tqdm 제어문자가 남지 않도록
# --log-file  : 별도 다운로드 로그 파일
python utils/dataset_download.py \
    --data-root "$DATA_ROOT"      \
    --workers   8                 \
    --retries   5                 \
    --no-progress                 \
    --log-file  "logs/download_${SLURM_JOB_ID}.log"

EXIT_CODE=$?

echo "=========================================="
echo "End        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Exit code  : $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
