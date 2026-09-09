#!/bin/bash
#SBATCH --job-name=PVT-M2-clusterpool
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m2_clusterpool_final_recipe_kfold_array_%a.log

# 2026-09-09: M1과 동일한 이유(논문 M1~M7 ablation 테이블 최종 레시피 재작성) — M2(WSI+Clinical,
# RNA 없음)에도 ClusterPool(models/vit_m1.py, M2가 상속)을 이식해 M4와 pooling 메커니즘을
# 통일한다. RNA가 없어 mutation/gene-set은 해당 없음 — clinical(staging+margin)+CLR100은 M4와
# 동일하게 유지.
#
# 2seed(84,126) x 5fold.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M2_uni2native_STG_R_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model M2_uni2native_STG_R_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1 --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 실제 생성된 CSV로 확인 권장: `ls .logs/kfold_preds/tcga_M2_uni2native*CLUSTERPOOL*`)
#
# 제출: sbatch sbatch/m2_clusterpool_final_recipe_2seed_kfold_array_hpc.sh

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

log=".logs/train_tcga_m2_clusterpool_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M2 ClusterPool(확정 레시피) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M2 --cluster-pool --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-staging --clinical-margin --clinical-lr-mult 100 \
    --patch-keep-frac 0.8 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0909m2_clusterpool_final_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== M2 ClusterPool(확정 레시피) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
