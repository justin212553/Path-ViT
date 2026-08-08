#!/bin/bash
#SBATCH --job-name=PVT-PMA-uni2-coxadd-stg-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_uni2_coxadd_stg_multiseed_kfold_array_%a.log

# pma_uni2_coxadd_stg_kfold_array_hpc.sh(단일 seed, 5-fold)의 3-seed 확장판.
#
# internal test c-index가 fold별로 워낙 크게 흔들려서(단일 seed 기준 std=0.086, 범위 0.51~0.74)
# 이 프로젝트 표준 3시드(42/84/126)로 5-fold 전체를 반복해, scripts/pool_multiseed_kfold_preds.py로
# (a) seed 간 pooled c-index 분산("이 정도 흔들린다"는 불확실성 폭)과 (b) 환자 단위 예측 평균
# 앙상블(그 흔들림을 실제로 줄인 더 뾰족한 최종 추정치)을 함께 낸다. 환자 한 명은 세 seed 각각에서
# 서로 다른(그 환자를 한 번도 학습에 안 쓴) fold 모델의 예측을 받으므로, 이 세 예측을 평균해도
# held-out 원칙은 깨지지 않는다(같은 seed의 5개 checkpoint를 그대로 평균하는 것과 다른 점 — 그건
# 한 환자가 4/5 fold의 train/val에 이미 들어가 있어서 누출이 생긴다).
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5 로 변환해 15개(3 seed x 5 fold) 독립
# job으로 제출한다. free-gpu가 한산하면 다 같이 뜨고, 아니면 자리 나는 대로 채워서 돌아가니 순서와
# 무관하게 안전하다(다른 kfold array 스크립트와 동일 관례).
#
# 로그 파일명에 실제 --seed 값을 그대로 반영한다 — 예전 pma_uni2_coxadd_stg_kfold_array_hpc.sh에서
# --seed만 수동으로 바꾸고 로그 파일명 문자열은 안 바꿔서 헷갈렸던 문제(2026-08-08) 재발 방지.
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_uni2_coxadd_stg_multiseed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== PMA(uni2,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0808pma_uni2_coxadd_stg_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA(uni2,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
