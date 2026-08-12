#!/bin/bash
#SBATCH --job-name=PVT-PMA-uni-coxadd-stg-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_uni_coxadd_stg_multiseed_kfold_array_%a.log

# pma_uni2_coxadd_stg_multiseed_kfold_array_hpc.sh(UNI2 baseline)의 UNI(v1) 버전 —
# scripts/train_pancancer_paad_brca.py(PAAD+BRCA 공동학습, UNI v1)와 진짜 같은 조건으로
# 비교하기 위한 대조군. 2026-08-11: 공동학습 결과(internal 0.6581/external 0.6168, 3seed
# 앙상블)가 UNI2 baseline(internal 0.6359/external 0.6337) 대비로는 internal 개선/external
# 하락을 보였는데, backbone(uni vs uni2) 차이가 섞여 있어 깨끗한 비교가 아니었다 — PAAD 단독
# UNI(v1) baseline은 지금까지 seed=84 5-fold(internal만, external 없음)로만 존재해서, 이걸
# 3seed x 5fold + external까지 마저 채운다.
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5 (다른 multiseed array와 동일 관례).
#
# 완료 후(15개 fold 로그 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
# (그 다음 pma_uni_coxadd_stg_multiseed_external_eval_hpc.sh로 external도 채울 것)
#
# 제출: sbatch sbatch/pma_uni_coxadd_stg_multiseed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PMA_uni_INT1500_SS_AUX_STG_R_DISP_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== PMA(uni,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0811pma_uni_coxadd_stg_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA(uni,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
