#!/bin/bash
# PMA_uni2official_INT1500_SS_AUX_STG_R_DISP_COX_ADD — 3seed(42/84/126) x 5fold 전체를
# 로컬에서 순차로 돌린다. HPC에 자리가 안 나서 로컬로 전환(2026-08-12). seed42/fold0 파일럿이
# 약 5분 걸렸으니 15개 전부 순차로 돌면 대략 1시간~1시간반 예상.
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 84 126)
N_FOLDS=5

for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_PMA_uni2official_INT1500_SS_AUX_STG_R_DISP_COX_ADD_kfold5_fold${FOLD}.log"
    echo "=== PMA(uni2official,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
    python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --backbone uni2official \
        --clinical-margin --clinical-staging --combine-mode cox_add \
        --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0812pma_uni2official_multiseed_kfold5_local 2>&1 | tee "${log}"
    echo "=== PMA(uni2official,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
  done
done
