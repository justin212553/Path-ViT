#!/bin/bash
#SBATCH --job-name=PVT-PMA-uni2-lr1e4wd1e2-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_uni2_coxadd_stg_lr1e4_wd1e2_multiseed_kfold_array_%a.log

# lr/weight_decay 스윕(pma_uni2_coxadd_stg_lr_wd_sweep_array_hpc.sh, 12조합 단일 6:2:2 split)
# 결과, lr=1e-4/wd=1e-2 조합이 val_c_index=0.6533·test_c_index=0.6585로 12개 중 최고였다
# (baseline lr=1e-5/wd=1e-1: val=0.6460/test=0.6484). 목표가 internal test c-index를 올리는
# 것이므로, 이 조합을 baseline(pma_uni2_coxadd_stg_multiseed_kfold_array_hpc.sh)과 동일한
# 3시드(42/84/126)x5-fold 프로토콜로 검증한다 — 단일 split 결과는 이 프로젝트에서 반복해서
# 재현 안 된 전례가 많아(예: NOTOP 구조적 ablation, M1_POOL 단일시드 결과) 멀티시드 확인 없이는
# 채택하지 않는다.
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5 (baseline과 동일한 15-job 관례).
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_LR1e-04_WD1e-02 \
#       --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_uni2_coxadd_stg_lr1e4_wd1e2_multiseed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

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

log=".logs/train_tcga_seed${SEED}_PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_LR1e-04_WD1e-02_kfold5_fold${FOLD}.log"

echo "=== PMA(uni2,cox_add,STG+R,lr=1e-4,wd=1e-2) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --lr 1e-4 --weight-decay 1e-2 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0810pma_uni2_coxadd_stg_lr1e4_wd1e2_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA(uni2,cox_add,STG+R,lr=1e-4,wd=1e-2) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
