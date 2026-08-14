#!/bin/bash
# M6(RNA 단독)_RNASNN — RNAEncoder를 PORPOISE SNN_Block 스타일(ELU+AlphaDropout)로 바꾼 변형을
# 3seed(42/84/126) x 5fold 전체로 검증한다. RNA만 쓰는 모델이라 WSI 처리가 없어 매우 빠르다.
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 84 126)
N_FOLDS=5

for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_M6_INT1500_RNASNN_kfold5_fold${FOLD}.log"
    echo "=== M6(RNASNN) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
    python -u ./train.py --M6 --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --rna-snn \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0813_m6_rnasnn_multiseed_kfold5_local 2>&1 | tee "${log}"
    echo "=== M6(RNASNN) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
  done
done
