#!/bin/bash
# M1_uni2_DISP_NOVIT_COORD — M1(WSI 단독, self-ABMIL, skip-patch-vit, DISP, aux-free)에
# --coord-embed(학습 파라미터 없는 sinusoidal 위치 인코딩, models/vit_encoder.py::
# SpatialPositionEmbedding을 patch_tokens에 잔차로 추가) 추가.
# fold0 파일럿: best_epoch 1->7, best_val_c_index 0.3820->0.4236, test_c_index 0.4137->0.4518.
set -e
cd "$(dirname "$0")/.."

SEED=42
N_FOLDS=5

for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_M1_uni2_DISP_NOVIT_COORD_kfold5_fold${FOLD}.log"
    echo "=== M1+coord-embed(uni2,skip-patch-vit,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
    python -u ./train.py --M1 --skip-patch-vit --attn-dispersion --coord-embed \
        --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m1_coordembed_seed42_kfold5_local 2>&1 | tee "${log}"
    echo "=== M1+coord-embed(uni2,skip-patch-vit,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
done
