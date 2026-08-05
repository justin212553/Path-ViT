#!/bin/bash
#SBATCH --job-name=PVT-M3-tcga
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_m3.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 2026-08-04: train_m1_hpc.sh와 동일한 이유(컴퓨트 노드 인터넷 문제로 wandb.log()가 멈추는 것 대응)
export WANDB_MODE=offline

# train_m1_hpc.sh와 동일한 이유(재현성 격리를 위해 단독 실행) — M3(WSI+RNA, clinical 제외,
# ViT_PMA --no-clinical)만 단독 학습. M3/PMA는 RNA가 있으니 SS+AUX+AUG+DISP 전부 적용.
#
# --tile-decode-workers 8 / --cache-val-tiles: train_m1_hpc.sh와 동일한 이유(epoch 1도 못 넘기던
# 진짜 병목 — 타일 디코딩 스레드풀 하드코딩 + evaluate()가 train_eval/val을 캐시 없이 디스크에서
# 매 epoch 새로 디코딩).
python -u ./train.py --PMA --no-clinical --rna-genes literature_1500 --dataset tcga --external --seed 84 \
    --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles "$@"
