#!/bin/bash
#SBATCH --job-name=PVT-FINAL-M4PMA-3seed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/final_m4_pma_3seed_kfold_array_%a.log

# 최종 확정용 M4/PMA(WSI+Clinic+RNA, co-attention+MLP+aux+concat+cox_add) 3seed x 5fold 재현.
# 현재 코드베이스(stage-stratify 반영 이후) 기준 — paper/results_table_pma_family_3seed_kfold_ci.md의
# pre-stratify 값(internal 0.6461[2seed]/external 0.6337)을 대체할, 논문에 실제로 실릴 최종
# 수치. 원본 태그(PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD)를 그대로 재현.
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac --model PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/final_m1m4_3seed_kfold_hpc/m4_pma_3seed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(42 84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log="paper/.hpc/train_tcga_seed${SEED}_PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== FINAL M4/PMA(WSI+Clinic+RNA,cox_add) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --dataset tcga --external --seed "${SEED}" \
    --rna-genes literature_1500_intersection \
    --backbone uni2native \
    --clinical-staging --clinical-margin \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --combine-mode cox_add \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0816_final_m4pma_3seed_kfold_hpc 2>&1 | tee "${log}"
echo "=== FINAL M4/PMA(WSI+Clinic+RNA,cox_add) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
