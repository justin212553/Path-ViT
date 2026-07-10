#!/bin/bash
#SBATCH --job-name=PVT-T-cptac
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_cptac.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --fusion이면 SBATCH --output(train_cptac.log) 대신 별도 로그 파일로 재지정
for arg in "$@"; do
    if [ "$arg" = "--fusion" ]; then
        exec > /pub/wonseukl/Path-ViT/.logs/train_cptac_fusion.log 2>&1
        break
    fi
done

python -u ./train.py --dataset cptac "$@"
