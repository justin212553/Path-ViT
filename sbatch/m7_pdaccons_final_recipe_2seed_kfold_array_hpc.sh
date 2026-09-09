#!/bin/bash
#SBATCH --job-name=PVT-M7-pdaccons-final
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m7_pdaccons_final_recipe_kfold_array_%a.log

# 2026-09-09: 논문 M1~M7 ablation 테이블 최종 레시피 재작성 — M7(Clinical+RNA)을 기존
# M7_INT1500_STG_R_COX_ADD(literature_1500_intersection, CPTAC 리키지 있음)에서
# pdac_consistency_1500(문헌 소스, CPTAC/TCGA 전혀 미참조)+CNV+mutation+both-loss로 교체 —
# M4(=PMA ClusterPool)/M3/M6와 gene-set·loss를 전부 맞춘다.
#
# WSI가 없어 backbone 인자 자체가 없다.
#
# 2seed(84,126) x 5fold.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M7_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model M7_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 실제 생성된 CSV로 확인 권장: `ls .logs/kfold_preds/tcga_M7*NLLSURV*`)
#
# 제출: sbatch sbatch/m7_pdaccons_final_recipe_2seed_kfold_array_hpc.sh

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

log=".logs/train_tcga_m7_pdaccons_final_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M7(pdac_consistency_1500, 확정 레시피) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train_light.py --M7 --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
    --use-cnv --clinical-margin --clinical-staging --clinical-mutation --combine-mode cox_add \
    --clinical-lr-mult 100 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0909m7_pdaccons_final_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== M7(pdac_consistency_1500, 확정 레시피) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
