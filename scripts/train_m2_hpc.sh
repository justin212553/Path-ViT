#!/bin/bash
#SBATCH --job-name=PVT-M2-tcga
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_m2.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# train_m1_hpc.sh와 동일한 이유(재현성 격리를 위해 단독 실행) — M2(WSI+Clinical)만 단독 학습.
# SS+AUG+DISP 레시피, AUX는 RNA 없는 M2엔 대응 항목 없어 제외.
python -u ./train.py --M2 --dataset tcga --external --seed 42 \
    --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion "$@"
