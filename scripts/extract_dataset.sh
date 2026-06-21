#!/bin/bash
#SBATCH --job-name=extract_dataset
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=download_progress.log

cd /pub/wonseukl/Path-ViT/

source .venv/bin/activate
./.venv/bin/python -u utils/extract_data.py