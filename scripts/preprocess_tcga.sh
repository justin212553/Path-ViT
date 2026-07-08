#!/bin/bash
#SBATCH --job-name=preprocess_tcga
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/preprocess_tcga_%j.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

python -um data.preprocess --dataset tcga "$@"
