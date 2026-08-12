#!/bin/bash
# PMA_uni_INT1500_SS_AUX_STG_R_DISP_COX_ADD(PAAD 단독, UNI v1 baseline) 15개 checkpoint(3seed x
# 5fold, 10개는 HPC 학습분+오늘 5개 로컬 재학습분)로 재학습 없이 external(cptac)만 재평가 —
# sbatch/pma_uni_coxadd_stg_multiseed_external_eval_hpc.sh와 동일 원리, 로컬 실행판.
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 84 126)
N_FOLDS=5

for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    MATCHES=(models/checkpoint/survival_tcga_uni_seed${SEED}_*STG_R_DISP_COX_ADD_FOLD${FOLD}OF${N_FOLDS}_best_pma.pt)
    if [ ! -f "${MATCHES[0]}" ]; then
      echo "[SKIP] seed=${SEED} fold=${FOLD}: checkpoint를 못 찾음"
      continue
    fi
    if [ "${#MATCHES[@]}" -gt 1 ]; then
      echo "[경고] seed=${SEED} fold=${FOLD}: checkpoint가 ${#MATCHES[@]}개 매칭됨 — 첫 번째만 사용: ${MATCHES[0]}"
    fi
    CKPT="${MATCHES[0]}"

    echo "=== external eval-only: seed=${SEED} fold=${FOLD} ckpt=${CKPT} Start: $(date) ==="
    python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --backbone uni \
        --clinical-margin --clinical-staging --combine-mode cox_add \
        --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" \
        --eval-external-ckpt "${CKPT}"
    echo "=== external eval-only: seed=${SEED} fold=${FOLD} Complete: $(date) ==="
  done
done
