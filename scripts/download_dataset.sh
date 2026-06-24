#!/bin/bash
#SBATCH --job-name=download
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/download_progress.log

cd /pub/wonseukl/Path-ViT/

PATIENTS="${1:?patient 범위를 인자로 지정하세요. 예: sbatch download_dataset.sh 0-49}"

./.venv/bin/python -u utils/dataset_download_zip.py --patients "$PATIENTS"