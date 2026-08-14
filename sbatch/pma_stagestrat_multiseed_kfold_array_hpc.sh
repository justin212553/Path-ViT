#!/bin/bash
#SBATCH --job-name=PVT-PMA-stagestrat-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_stagestrat_multiseed_kfold_array_%a.log

# pma_uni2_coxadd_stg_multiseed_kfold_array_hpc.sh(기존 baseline)에 --stage-stratify만 추가한
# 대조실험. 2026-08-14 조사에서 fold별 internal log-rank p가 요동친 원인이 event 비율/표본
# 크기가 아니라 stage 구성 쏠림(나쁜 fold는 Stage IIB가 77%까지 쏠림)으로 보여, split
# stratification key에 ajcc_stage를 추가했다(data/dataset.py::WSISurvivalDataset
# use_stage_stratify, 기본 False — 이 실험에서만 켬).
#
# 주의: --stage-stratify를 켜면 같은 seed라도 fold 배정(어떤 환자가 어느 fold에 들어가는지)
# 자체가 기존과 달라진다 — 즉 이 결과는 기존 baseline과 "같은 데이터 split, stratify 유무만
# 다름"이 아니라 "완전히 다른 split"과의 비교다. log-rank p의 fold간 변동폭이 줄어드는지가
# 1차 관심사이고, c-index 자체의 우열은 참고만 할 것(split이 다르면 직접 비교가 정확하지 않음).
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_STGSTRAT --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_stagestrat_multiseed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_STGSTRAT_kfold5_fold${FOLD}.log"

echo "=== PMA(uni2,cox_add,STG+R,stage-stratify) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --stage-stratify \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814pma_stagestrat_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA(uni2,cox_add,STG+R,stage-stratify) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
