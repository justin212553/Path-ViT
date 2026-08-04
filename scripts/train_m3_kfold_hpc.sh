#!/bin/bash
#SBATCH --job-name=PVT-M3-kfold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m3_kfold_array_%a.log

# train_m3_hpc.sh(단일 6:2:2 split)의 K-fold 버전 — train_m1_kfold_hpc.sh와 동일한 패턴/이유.
# M3(WSI+RNA, clinical 제외, ViT_PMA --no-clinical) — RNA가 있으니 EX+SS+AUX+AUG+DISP 전부.
#
# 완료 후 풀링(model_prefix에 --no-clinical의 _NOCLINICAL 태그가 들어간다는 점 주의):
#   python scripts/pool_kfold_preds.py --dataset tcga --model PMA_EX_SS_AUX_AUG_NOCLINICAL_DISP --seed 42 --n-folds 5
#
# 제출: sbatch scripts/train_m3_kfold_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

FOLD=$SLURM_ARRAY_TASK_ID
log=".logs/train_tcga_seed42_PMA_EX_SS_AUX_AUG_NOCLINICAL_DISP_kfold5_fold${FOLD}.log"

echo "=== M3(PMA_EX_SS_AUX_AUG_NOCLINICAL_DISP) fold=${FOLD}/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --no-clinical --rna-genes literature_1500 --dataset tcga --external --seed 42 \
    --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
    --fold "${FOLD}" --n-folds 5 --group-ts 0804m3_kfold5_array 2>&1 | tee "${log}"
echo "=== M3(PMA_EX_SS_AUX_AUG_NOCLINICAL_DISP) fold=${FOLD}/5 Complete: $(date) ==="
