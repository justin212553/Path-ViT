#!/bin/bash
# M1_uni2_DISP_NOVIT_COORD_CAT — coord-embed의 concat+fusion 변형(잔차 add 대신
# [patch_tokens ‖ coord_embed] -> Linear->LayerNorm->GELU로 다시 embed_dim에 투영).
# fold0 파일럿: best_epoch 7->9, best_val_c_index 0.4236->0.6247, test_c_index 0.4518->0.6413
# (다만 HR CI [0.66,4.10]/logrank p=0.134로 유의하지 않아 fold0 단일값 과대평가 가능성 있음 —
# 5-fold 전체로 검증).
set -e
cd "$(dirname "$0")/.."

SEED=42
N_FOLDS=5

for ((FOLD=1; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_M1_uni2_DISP_NOVIT_COORD_CAT_kfold5_fold${FOLD}.log"
    echo "=== M1+coord-embed-concat(uni2,skip-patch-vit,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
    python -u ./train.py --M1 --skip-patch-vit --attn-dispersion --coord-embed --coord-embed-concat \
        --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m1_coordcat_seed42_kfold5_local 2>&1 | tee "${log}"
    echo "=== M1+coord-embed-concat(uni2,skip-patch-vit,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
done
