#!/bin/bash
#SBATCH --job-name=tune
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                  
#SBATCH --time=18:00:00            
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/tune.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u ./tune.py