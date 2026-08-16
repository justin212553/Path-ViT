#!/bin/bash
# M2_uni2_STG_R_DISP_NOVIT_XMLP (combine-mode=concat, wsi-extra-mlp만, 좌표 없음) — fold0
# 파일럿: best_val_c=0.6173, test_c_index=0.5298 (참고: cox_add-no-mlp baseline=0.5860).
set -e
cd "$(dirname "$0")/.."

SEED=42
N_FOLDS=5

for ((FOLD=1; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_M2_uni2_STG_R_DISP_NOVIT_XMLP_kfold5_fold${FOLD}.log"
    echo "=== M2 concat+wsi-extra-mlp seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
    python -u ./train.py --M2 --skip-patch-vit --attn-dispersion --clinical-margin --clinical-staging --wsi-extra-mlp \
        --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m2_xmlp_concat_seed42_kfold5_local 2>&1 | tee "${log}"
    echo "=== M2 concat+wsi-extra-mlp seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
done
