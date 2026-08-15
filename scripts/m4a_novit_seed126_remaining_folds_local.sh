#!/bin/bash
# M4A_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT — seed126(M4-NOVIT에서 가장 부진했던 시드)의
# 나머지 fold(1~4)를 마저 돌린다. fold0은 이미 파일럿으로 완료(internal=0.6973, external=0.6099).
set -e
cd "$(dirname "$0")/.."

SEED=126
N_FOLDS=5

for ((FOLD=1; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_M4A_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT_kfold5_fold${FOLD}.log"
    echo "=== M4A+skip-patch-vit(uni2,cox_add,STG+R,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
    python -u ./train.py --M4A --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --clinical-margin --clinical-staging --combine-mode cox_add \
        --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
        --skip-patch-vit \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814m4a_novit_seed126_pilot_local 2>&1 | tee "${log}"
    echo "=== M4A+skip-patch-vit(uni2,cox_add,STG+R,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
done
