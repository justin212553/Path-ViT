#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/extract_features.log

cd /pub/wonseukl/Path-ViT/

conda init

conda activate Path-ViT

./.venv/bin/python -u data/extract_features.py