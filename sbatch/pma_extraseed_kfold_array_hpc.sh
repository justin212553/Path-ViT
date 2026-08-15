#!/bin/bash
#SBATCH --job-name=PVT-PMA-extraseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_extraseed_kfold_array_%a.log

# pma_uni2_coxadd_stg_multiseed_kfold_array_hpc.sh(seed 42/84/126)의 추가 시드 3개(168/210/252)
# 버전 — 2026-08-14: seed126이 여러 WSI 아키텍처 실험에서 반복적으로 다른 두 시드보다 낮게
# 나온 것이 진짜 이상치인지, 아니면 그냥 3개뿐인 표본에서 흔히 있는 흔들림인지 확인하기 위해
# baseline PMA 자체를 시드 3개 더(총 6개) 돌린다. 168/210은 --full-train noise floor 실험
# (2026-07-26)에서 이미 쓴 값이라 비교 연속성이 있고, 252는 새 값(42의 배수 취향).
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 168,210,252 --n-folds 5 --bootstrap 2000
#   (기존 42,84,126과 합쳐 6-seed로 보려면 pool 스크립트의 --seeds에 6개를 다 나열)
#
# 제출: sbatch sbatch/pma_extraseed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== PMA(uni2,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814pma_extraseed_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA(uni2,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
