#!/bin/bash
#SBATCH --job-name=PVT-M5-final
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m5_final_recipe_kfold_array_%a.log

# 2026-09-09: 논문 M1~M7 ablation 테이블 최종 레시피 재작성 — M5(Clinical only)에 both-loss를
# 새로 이식(models/clinical_only.py에 surv_n_classes 추가, train_light.py에 --surv-loss/
# fit_survival_bins 배선, 2026-09-09)해서 M6/M7과 loss를 맞춘다 — 이래야 M5->M7(RNA 추가) delta가
# RNA 효과만 순수하게 반영한다(사용자 결정, loss 변경 효과가 섞이지 않게).
#
# WSI가 없어 backbone 인자 자체가 없다.
#
# 2seed(84,126) x 5fold.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M5_STG_R_NLLSURV4_NLLCOX1 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model M5_STG_R_NLLSURV4_NLLCOX1 --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 실제 생성된 CSV로 확인 권장: `ls .logs/kfold_preds/tcga_M5*NLLSURV*`)
#
# 제출: sbatch sbatch/m5_final_recipe_2seed_kfold_array_hpc.sh

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

log=".logs/train_tcga_m5_final_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M5(확정 레시피) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train_light.py --M5 --raw-linear --dataset tcga --external --seed "${SEED}" \
    --clinical-margin --clinical-staging \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0909m5_final_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== M5(확정 레시피) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
