#!/bin/bash
# PMA_uni2 baseline, fold0 고정, 3seed(42/84/126) 각각 SWAD로 학습해서 SWAD 평균 가중치를
# checkpoint로 저장한다(--eval-soup-ckpts로 나중에 3개를 다시 souping하기 위함).
# seed42는 이전 파일럿에서 이미 한 번 돌렸지만 그때는 SWAD 체크포인트 저장 로직이 없어서
# 다시 돌린다.
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 84 126)
FOLD=0
N_FOLDS=5

for SEED in "${SEEDS[@]}"; do
  log=".logs/swad_pilot_seed${SEED}_fold${FOLD}.log"
  echo "=== SWAD pilot: seed=${SEED} fold=${FOLD} Start: $(date) ==="
  python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
      --backbone uni2 \
      --clinical-margin --clinical-staging --combine-mode cox_add \
      --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
      --fold "${FOLD}" --n-folds "${N_FOLDS}" --swa --swad --swad-tolerance 0.02 \
      --group-ts 0812_swad_3seed_fold0 2>&1 | tee "${log}"
  echo "=== SWAD pilot: seed=${SEED} fold=${FOLD} Complete: $(date) ==="
done
