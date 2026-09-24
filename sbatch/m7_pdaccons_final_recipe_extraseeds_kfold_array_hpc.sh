#!/bin/bash
#SBATCH --job-name=PVT-M7-pdaccons-final-extraseeds
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-14
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m7_pdaccons_final_recipe_extraseeds_kfold_array_%a.log

# 2026-09-24: pma_clusterpool_pdaccons_final_recipe_extraseeds_kfold_array_hpc.sh(M4)와 짝을
# 이루는 M7 5-seed 확장 — 헤더 설명은 그 파일 참조. 레시피는
# sbatch/m7_pdaccons_final_recipe_2seed_kfold_array_hpc.sh(M7, 확정 레시피)와 완전히 동일 —
# SEEDS 배열과 --array 범위만 다르다.
#
# WSI가 없어 backbone 인자 자체가 없다(train_light.py).
#
# 3seed(42,168,210) x 5fold = 15 array task.
#
# 완료 후(기존 84/126과 합쳐 5seed로 풀링):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M7_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,42,168,210 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model M7_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,42,168,210 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m7_pdaccons_final_recipe_extraseeds_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(42 168 210)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_m7_pdaccons_final_extraseeds_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M7(pdac_consistency_1500, 확정 레시피, extra seed) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train_light.py --M7 --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
    --use-cnv --clinical-margin --clinical-staging --clinical-mutation --combine-mode cox_add \
    --clinical-lr-mult 100 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0924m7_pdaccons_final_extraseeds_kfold5_array 2>&1 | tee "${log}"
echo "=== M7(pdac_consistency_1500, 확정 레시피, extra seed) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
