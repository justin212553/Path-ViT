#!/bin/bash
#SBATCH --job-name=preprocess_tcga
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/preprocess_tcga.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

python -um data.preprocess --dataset tcga "$@"
