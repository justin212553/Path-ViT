#!/bin/bash
#SBATCH --job-name=PVT-PMA-clusterpool-pdaccons-extraseeds
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_clusterpool_pdaccons_final_recipe_extraseeds_kfold_array_%a.log

# 2026-09-24: 논문 리뷰 피드백 — "시드 2개"는 논문 스스로가 논증하는 "단일 시드로는 신뢰 못
# 한다"(Section 4.5, The Case for Multi-Seed Evaluation)는 주장을 자기 모델(M4/M7)에는 3~5개가
# 아니라 2개만 적용한 자기모순이라는 지적. M4에 대해 시드 3개(42, 168, 210)를 추가로 돌려
# 84/126과 합쳐 총 5시드로 만든다. 레시피는
# sbatch/pma_clusterpool_pdaccons_final_recipe_2seed_kfold_array_hpc.sh(M4, 확정 레시피)와
# 완전히 동일 — SEEDS 배열과 --array 범위만 다르다(기존 84/126 결과는 그대로 재사용, 덮어쓰지
# 않음).
#
# 3seed(42,168,210) x 5fold = 15 array task.
#
# 완료 후(기존 84/126과 합쳐 5seed로 풀링):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,42,168,210 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,42,168,210 --n-folds 5 --bootstrap 2000
# seed별/fold별 C-index 분포 boxplot용 원자료는 --seeds를 쉼표로 나눠 개별 풀링하거나
# .logs/kfold_preds/의 개별 fold CSV에서 직접 뽑을 것.
#
# 제출: sbatch sbatch/pma_clusterpool_pdaccons_final_recipe_extraseeds_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1_kfold5_fold${FOLD}.log"

echo "=== PMA ClusterPool(pdac_consistency_1500, 확정 레시피, extra seed) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --cluster-pool --combine-mode cox_add --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0924pma_clusterpool_pdaccons_final_extraseeds_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA ClusterPool(pdac_consistency_1500, 확정 레시피, extra seed) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
