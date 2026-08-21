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

# M2_POOL baseline(m2_pool_3seed_kfold_array_hpc.sh)을 M4/PMA와 clinical 정보량(margin+staging)
# 기준으로 동일선상에 놓기 위해 --clinical-margin --clinical-staging --combine-mode cox_add를
# 추가(2026-08-17 사용자 지적: "M4는 clinical을 강화해놓고 M2는 age/sex만 쓰면 동일 비교선상이
# 아니다").
#
# 2026-08-21: pooling_mode를 coattn(clinical이 4개 pooling 관점의 co-attention query, M4의
# RNA-query와 대칭)에서 selfattn(clinical이 pooling에 전혀 관여하지 않음, models/vit_m1_pool.py와
# 동일 모듈 재사용)으로 변경 — clinical cox_add를 ClinicalEncoder(MLP) 경유로 바꿨다가 M7
# ablation에서 internal -0.025로 확인되어 raw feature 직결로 원복하는 김에(models/vit_pma.py,
# models/clinical_rna_only.py와 동일 결론), M2는 애초에 M3/M4를 이길 일이 없는 baseline/floor
# 모델이므로 복잡한 co-attention 구조 대신 가장 단순한 조합(selfattn pooling + cox_add raw
# feature, 이 조합에선 ClinicalEncoder 자체가 생성되지 않음)으로 확정(사용자 결정: "성능이
# 폭발적으로 오른다 해도 M3/M4에는 못 비빌태니 차라리 낫고").
# 태그: M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD_SELFATTN (train.py model_prefix 부착 순서 그대로 —
# _SS는 patch-keep-frac 0.8 표시이지 pooling_mode와 무관, _SELFATTN이 맨 뒤에 pooling_mode=selfattn을 표시).
#
# --rna-aux-weight는 WSI mean-pool embedding에서 RNA를 예측하는 구조라(models/rna_predictor.py
# ::RNAPredictionHead.forward(wsi_meanpool_embed)) M2(RNA 자체가 없음)엔 애초에 해당 없음 —
# 추가 안 함.
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD_SELFATTN --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac --model M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD_SELFATTN --seeds 84,126 --n-folds 5 --bootstrap 2000
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

log="paper/.hpc/train_tcga_seed${SEED}_M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD_SELFATTN_kfold5_fold${FOLD}.log"

echo "=== FINAL M2_POOL+fullclin(selfattn,cox_add-raw,WSI+Clinic[age/sex+margin+staging]) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M2_POOL --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --attn-dispersion \
    --patch-keep-frac 0.8 \
    --clinical-margin --clinical-staging \
    --pooling-mode selfattn \
    --combine-mode cox_add \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0821_final_m2pool_selfattn_coxadd_3seed_kfold_hpc 2>&1 | tee "${log}"
echo "=== FINAL M2_POOL+fullclin(selfattn,cox_add-raw,WSI+Clinic[age/sex+margin+staging]) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
