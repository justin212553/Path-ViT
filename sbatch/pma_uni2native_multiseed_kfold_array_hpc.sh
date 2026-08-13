#!/bin/bash
#SBATCH --job-name=PVT-PMA-uni2native-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_uni2native_multiseed_kfold_array_%a.log

# pma_uni2official_multiseed_kfold_array_hpc.sh의 후속 — backbone을 uni2native로 바꿔
# uni2official에서 확인된 두 confound(DX 슬라이드만 포함, coords 스케일 ~4000배 불일치)를
# 피한 상태에서 "해상도(0.5MPP)" 변수 하나만 격리해 재검증한다.
#
# 사전 단계(이 job 제출 전에 반드시 순서대로 완료):
#   1) sbatch sbatch/preprocess_uni2native_retile_array_hpc.sh (16-shard 재타일링)
#   2) sbatch sbatch/extract_features_uni2native_array_hpc.sh (tcga/cptac UNI2-h 추출)
#   3) python scripts/reconcile_uni2native_features.py (기존 patches 트리에 features_uni2native.pt/
#      coords_uni2native.pt 정리)
#
# uni2official 3seed x 5fold 결과(--epochs 100 --early-stop-patience 10, 참고용):
# internal 0.6359->0.5817(-0.054), external 0.6337->0.6304(-0.003, 사실상 동일) — 다만 슬라이드
# 구성(DX만)과 coords 스케일 버그가 섞여있어 해상도 자체의 순수 효과는 이 실험(uni2native)으로
# 다시 봐야 한다.
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD_ES10 --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_uni2native_multiseed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD_ES10_kfold5_fold${FOLD}.log"

echo "=== PMA(uni2native,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --epochs 100 --early-stop-patience 10 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0812pma_uni2native_ep100es_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA(uni2native,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
