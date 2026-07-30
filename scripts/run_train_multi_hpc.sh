#!/bin/bash
#SBATCH --job-name=PVT-M-tcga
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_multi.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 128GB RAM이면 로컬(32GB, maxsize 24576≈18GB)·Kaggle(30GB, maxsize 30000≈22GB)보다
# 훨씬 여유롭다 — train split 전체 패치(~29,575개)를 다 담고도 넉넉하게 40000으로 잡는다
# (40000 * ~0.75MB ≈ 30GB, 128GB 중 나머지는 OS·모델 4개·optimizer state·CUDA 컨텍스트용).
python -u ./train_multi.py --M1 --M2 --M3 --PMA --dataset tcga --external --tile-augment --seed 42 --tile-cache-maxsize 40000 "$@"
