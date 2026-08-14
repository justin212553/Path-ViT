#!/bin/bash
# PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_TTE — 종양/정상/미상 슬라이드 타입을 학습 가능한
# 임베딩으로 ViT 입력에 주입하는 --tumor-type-embed를 3seed(42/84/126) x 5fold 전체로 검증한다.
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 84 126)
N_FOLDS=5

for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_TTE_kfold5_fold${FOLD}.log"
    echo "=== PMA(uni2,cox_add,STG+R,TTE) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
    python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --clinical-margin --clinical-staging --combine-mode cox_add \
        --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
        --tumor-type-embed \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0813_pma_tte_multiseed_kfold5_local 2>&1 | tee "${log}"
    echo "=== PMA(uni2,cox_add,STG+R,TTE) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
  done
done
