#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/dataset.log

cd /pub/wonseukl/Path-ViT/

./.venv/bin/python -u utils/dataset_download_zip.py