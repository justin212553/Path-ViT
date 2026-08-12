#!/bin/bash
# PMA_uni_INT1500_SS_AUX_STG_R_DISP_COX_ADD(PAAD 단독, UNI v1 baseline) 3seed x 5fold 중
# HPC에서 중간에 끊긴 5개(seed42 fold0/2/3/4, seed84 fold0)만 로컬에서 마저 돌린다.
# seed126(전체)·seed42 fold1·seed84 fold1~4는 이미 HPC에서 완료돼 .logs/kfold_preds/에 있음.
set -e
cd "$(dirname "$0")/.."

# (SEED, FOLD) 쌍만 지정 — 남은 5개만.
COMBOS=("42 0" "42 2" "42 3" "42 4" "84 0")
N_FOLDS=5

for combo in "${COMBOS[@]}"; do
  read -r SEED FOLD <<< "$combo"
  log=".logs/train_tcga_seed${SEED}_PMA_uni_INT1500_SS_AUX_STG_R_DISP_COX_ADD_kfold5_fold${FOLD}.log"
  echo "=== PMA(uni,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
  python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
      --backbone uni \
      --clinical-margin --clinical-staging --combine-mode cox_add \
      --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
      --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0812pma_uni_coxadd_stg_missing_folds_local 2>&1 | tee "${log}"
  echo "=== PMA(uni,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
done
