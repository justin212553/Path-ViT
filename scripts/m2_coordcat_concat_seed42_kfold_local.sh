#!/bin/bash
# M2_uni2_STG_R_DISP_NOVIT_COORD_CAT (combine-mode=concat, cox_add 아님) — coord-embed-concat이
# cox_add에서 clinical_linear를 죽이는 문제(fold0/fold1 재현 확인)를 피해 concat 모드로
# 시도. fold0 파일럿: test_c_index=0.4702, zero-ablation에서 clinical이 죽지 않고 살아있음
# 확인(WSI/clinical 기여 비율 0.81).
set -e
cd "$(dirname "$0")/.."

SEED=42
N_FOLDS=5

for ((FOLD=1; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_M2_uni2_STG_R_DISP_NOVIT_COORD_CAT_kfold5_fold${FOLD}.log"
    echo "=== M2 concat+coord-embed-concat seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
    python -u ./train.py --M2 --skip-patch-vit --attn-dispersion --clinical-margin --clinical-staging \
        --coord-embed --coord-embed-concat \
        --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m2_coordcat_concat_seed42_kfold5_local 2>&1 | tee "${log}"
    echo "=== M2 concat+coord-embed-concat seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
done
