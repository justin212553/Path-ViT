#!/bin/bash
#SBATCH --job-name=PVT-FINAL-M2POOL-fullclin-3seed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/final_m2_pool_fullclin_3seed_kfold_array_%a.log

# M2_POOL baseline(m2_pool_3seed_kfold_array_hpc.sh)을 M4/PMA와 완전히 동일선상에 놓기 위해
# 세 가지를 추가(2026-08-17 사용자 지적: "M4는 clinical을 강화해놓고 M2는 age/sex만 쓰면
# 동일 비교선상이 아니다" + "M4 기준으로 나머지도 다 맞춰야 한다"):
#   1. --clinical-margin(절제연)  2. --clinical-staging(T/N/M/grade)
#   3. --combine-mode cox_add(clinical을 risk_head 스칼라에 classical Cox 가산항으로 결합 —
#      M4와 동일 방식, 기존 M2_POOL 기본값은 concat이었음)
# ViT_M2_Pool이 원래 staging을 구조적으로 미지원이라 models/vit_m2_pool.py(ClinicalEncoder/
# cox_add raw feature 양쪽에 use_staging 배선)와 train.py(모델 생성 시 stage_kwargs 전달 +
# combine_with_clinical_pool 분기의 stage_ord 계산 + --clinical-staging 허용 모델 목록)를
# 확장했다. pooling_mode(coattn, clinical이 4개 관점의 query — M4의 RNA-query co-attention과
# 구조적으로 대칭)는 그대로 유지 — 이건 이미 M4와 같은 메커니즘이라 바꿀 필요 없음.
# 로컬 스모크테스트(uni2, fold0, margin+staging+cox_add 조합) 통과 확인됨,
# 태그는 M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD.
#
# --rna-aux-weight는 WSI mean-pool embedding에서 RNA를 예측하는 구조라(models/rna_predictor.py
# ::RNAPredictionHead.forward(wsi_meanpool_embed)) M2(RNA 자체가 없음)엔 애초에 해당 없음 —
# 추가 안 함.
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac --model M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/final_m1m4_3seed_kfold_hpc/m2_pool_margin_3seed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

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

log="paper/.hpc/train_tcga_seed${SEED}_M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== FINAL M2_POOL+fullclin(coattn,cox_add,WSI+Clinic[age/sex+margin+staging]) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M2_POOL --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --attn-dispersion \
    --patch-keep-frac 0.8 \
    --clinical-margin --clinical-staging \
    --combine-mode cox_add \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0817_final_m2pool_fullclin_3seed_kfold_hpc 2>&1 | tee "${log}"
echo "=== FINAL M2_POOL+fullclin(coattn,cox_add,WSI+Clinic[age/sex+margin+staging]) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
