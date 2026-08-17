#!/bin/bash
#SBATCH --job-name=PVT-FINAL-M2POOL-3seed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/final_m2_pool_3seed_kfold_array_%a.log

# 최종 확정용 M2(M1_POOL + clinical, 원본 태그와 동일하게 기본값 유지 — pooling_mode=coattn,
# combine_mode=concat, --clinical-margin 없음) 3seed x 5fold 재현. 현재 코드베이스(stage-stratify
# 반영 이후) 기준 — paper/results_table_pma_family_3seed_kfold_ci.md의 pre-stratify 값
# (0.5196/0.5199)을 대체할 최종 수치. 원본 태그(M2_POOL_uni2native_SS_DISP)를 그대로 재현하기 위해
# --pooling-mode/--combine-mode/--clinical-margin을 일부러 넣지 않았다(전부 기본값).
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M2_POOL_uni2native_SS_DISP --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac --model M2_POOL_uni2native_SS_DISP --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/final_m1m4_3seed_kfold_hpc/m2_pool_3seed_kfold_array_hpc.sh

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

log="paper/.hpc/train_tcga_seed${SEED}_M2_POOL_uni2native_SS_DISP_kfold5_fold${FOLD}.log"

echo "=== FINAL M2_POOL(coattn,concat,WSI+Clinic) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M2_POOL --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --attn-dispersion \
    --patch-keep-frac 0.8 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0816_final_m2pool_3seed_kfold_hpc 2>&1 | tee "${log}"
echo "=== FINAL M2_POOL(coattn,concat,WSI+Clinic) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
