#!/bin/bash
#SBATCH --job-name=PVT-M4-novit-extraseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m4_novit_extraseed_kfold_array_%a.log

# m4_novit_multiseed_kfold_array_hpc.sh(seed 42/84/126)의 추가 시드 3개(168/210/252) 버전.
# 2026-08-14: M4+skip-patch-vit(NOVIT)에서 seed126이 다른 두 시드보다 뚜렷이 낮게 나온 게
# (internal pooled: 42=0.6885, 84=0.6432, 126=0.5832) 진짜 이상치인지 확인하기 위해 시드
# 3개를 더 돌린다(baseline PMA도 같은 시드로 동시에 확인 — sbatch/pma_extraseed_kfold_array_hpc.sh).
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT --seeds 168,210,252 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m4_novit_extraseed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(168 210 252)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT_kfold5_fold${FOLD}.log"

echo "=== M4+skip-patch-vit(uni2,cox_add,STG+R,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4 --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --skip-patch-vit \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814m4_novit_extraseed_kfold5_array 2>&1 | tee "${log}"
echo "=== M4+skip-patch-vit(uni2,cox_add,STG+R,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
