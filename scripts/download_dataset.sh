#!/bin/bash
#SBATCH --job-name=camelyon_download
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=download_progress.log

# 1. 우리의 대용량 메인 기지로 진입
cd /pub/wonseukl/Path-ViT/

# 2. 가상환경 깨우기 (한 칸 상위 폴더인 프로젝트 루트의 .venv를 바라봅니다)
source /opt/apps/anaconda/2024.02/etc/profile.d/conda.sh
conda activate ./.venv

# 3. 데이터 다운로드 엔진 가동 (utils 폴더 내부의 스크립트 실행)
python utils/dataset_download.py