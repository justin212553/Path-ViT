#!/bin/bash
#SBATCH --job-name=download
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/download_progress.log

cd /pub/wonseukl/Path-ViT/

PATIENTS="${1:?patient 범위를 인자로 지정하세요. 예: sbatch download_dataset.sh 0-49}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

python -u utils/dataset_download_zip.py --patients "$PATIENTS"