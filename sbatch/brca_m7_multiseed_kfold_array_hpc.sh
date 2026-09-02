#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M7-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m7_multiseed_kfold_array_%a.log

# sbatch/brca_m4_multiseed_kfold_array_hpc.sh의 M7(ClinicalRNAOnly, WSI 없음) 대조군 짝 —
# M4와 반드시 동일 (seed, fold, n_folds, external-tss)로 돌려야 "같은 데이터 분할에서 M4가
# M7을 넘냐"는 비교가 성립한다. WSI가 없어 훨씬 가벼우므로(로컬 스모크 테스트 2epoch 14초)
# 시간/메모리 예산을 M4보다 작게 잡았다.
#
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5 (M4와 동일 관례 — 같은 IDX면 같은
# (seed, fold) 조합).
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_M7_TOP1500 --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/brca_m7_multiseed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_brca_m7_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M7 seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m7 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --external-tss none --group-ts 0901_brca_m7_multiseed_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M7 seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
