#!/bin/bash
# M2_uni2_STG_R_DISP_NOVIT_COORD_CAT (concat 모드) 5개 checkpoint로 재학습 없이 external(cptac)만 재평가.
set -e
cd "$(dirname "$0")/.."

SEED=42
N_FOLDS=5

for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    MATCHES=(models/checkpoint/survival_tcga_uni2_seed${SEED}_*M2_uni2_STG_R_DISP_NOVIT_COORD_CAT_FOLD${FOLD}OF${N_FOLDS}_best_clinical.pt)
    if [ ! -f "${MATCHES[0]}" ]; then
      echo "[SKIP] fold=${FOLD}: checkpoint를 못 찾음"
      continue
    fi
    CKPT="${MATCHES[0]}"

    echo "=== external eval-only: fold=${FOLD} ckpt=${CKPT} Start: $(date) ==="
    python -u ./train.py --M2 --skip-patch-vit --attn-dispersion --clinical-margin --clinical-staging \
        --coord-embed --coord-embed-concat \
        --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" \
        --eval-external-ckpt "${CKPT}"
    echo "=== external eval-only: fold=${FOLD} Complete: $(date) ==="
done
