#!/bin/bash
# M4A_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT seed126 5개 checkpoint로 external(cptac)만 재평가.
set -e
cd "$(dirname "$0")/.."

SEED=126
N_FOLDS=5

for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    MATCHES=(models/checkpoint/survival_tcga_uni2_seed${SEED}_*STG_R_DISP_COX_ADD_NOVIT_FOLD${FOLD}OF${N_FOLDS}_best_clinical_rna_coattn.pt)
    if [ ! -f "${MATCHES[0]}" ]; then
      echo "[SKIP] fold=${FOLD}: checkpoint를 못 찾음"
      continue
    fi
    CKPT="${MATCHES[0]}"
    echo "=== external eval-only: fold=${FOLD} ckpt=${CKPT} Start: $(date) ==="
    python -u ./train.py --M4A --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --clinical-margin --clinical-staging --combine-mode cox_add \
        --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
        --skip-patch-vit \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" \
        --eval-external-ckpt "${CKPT}"
    echo "=== external eval-only: fold=${FOLD} Complete: $(date) ==="
done
