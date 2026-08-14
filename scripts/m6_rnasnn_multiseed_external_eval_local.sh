#!/bin/bash
# M6_INT1500_RNASNN 15개 checkpoint로 재학습 없이 external(cptac)만 재평가.
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 84 126)
N_FOLDS=5

for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    MATCHES=(models/checkpoint/survival_tcga_seed${SEED}_*RNASNN*FOLD${FOLD}OF${N_FOLDS}_best_rna_only.pt)
    if [ ! -f "${MATCHES[0]}" ]; then
      echo "[SKIP] seed=${SEED} fold=${FOLD}: checkpoint를 못 찾음"
      continue
    fi
    if [ "${#MATCHES[@]}" -gt 1 ]; then
      echo "[경고] seed=${SEED} fold=${FOLD}: checkpoint가 ${#MATCHES[@]}개 매칭됨 — 첫 번째만 사용: ${MATCHES[0]}"
    fi
    CKPT="${MATCHES[0]}"

    echo "=== external eval-only: seed=${SEED} fold=${FOLD} ckpt=${CKPT} Start: $(date) ==="
    python -u ./train.py --M6 --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --rna-snn \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" \
        --eval-external-ckpt "${CKPT}"
    echo "=== external eval-only: seed=${SEED} fold=${FOLD} Complete: $(date) ==="
  done
done
