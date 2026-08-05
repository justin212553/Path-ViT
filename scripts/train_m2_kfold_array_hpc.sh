#!/bin/bash
#SBATCH --job-name=PVT-M2-kfold-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m2_kfold_array_%a.log

# train_m1_kfold_array_hpc.sh와 동일한 이유/패턴(야간 유휴 시간대 노려서 5-fold 병렬 시도).
# race condition 픽스 반영됨(data/patch_utils.py) — 병렬로 돌려도 안전.
# SS+AUG+DISP — EX/AUX는 RNA 브랜치가 없는 M2엔 대응 항목 없어 제외.
#
# 2026-08-05: --time을 6h->24h로 늘렸다 — M1 array 작업이 6h로 돌다 fold 도중 타임아웃으로
# 끊긴 걸 확인(사용자 지적), array 태스크도 순차 스크립트와 동일한 24h 안전 마진을 준다.
#
# 완료 후 집계:
#   python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model M2_SS_AUG_DISP
#
# 제출: sbatch scripts/train_m2_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

FOLD=$SLURM_ARRAY_TASK_ID
log=".logs/train_tcga_seed84_M2_SS_AUG_DISP_kfold5_fold${FOLD}.log"

echo "=== M2_SS_AUG_DISP fold=${FOLD}/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M2 --dataset tcga --external --seed 84 \
    --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion \
    --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
    --fold "${FOLD}" --n-folds 5 --group-ts 0804m2_kfold5_array 2>&1 | tee "${log}"
echo "=== M2_SS_AUG_DISP fold=${FOLD}/5 Complete: $(date) ==="
