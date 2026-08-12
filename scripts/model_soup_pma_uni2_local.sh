#!/bin/bash
# PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD baseline — fold별로 3seed(42/84/126) 체크포인트의
# 가중치를 평균(model soup)해서 재학습 없이 internal(그 fold held-out)/external을 평가한다.
# train.py --eval-soup-ckpts 사용. 사용자 요청: SAM/SWA 외 시드 분산 완화 방법 탐색.
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 84 126)
N_FOLDS=5
CKPT_DIR="models/checkpoint"

for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
  CKPTS=""
  for SEED in "${SEEDS[@]}"; do
    CKPT="${CKPT_DIR}/survival_tcga_uni2_seed${SEED}_INT1500_SS_AUX_STG_R_PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_FOLD${FOLD}OF${N_FOLDS}_best_pma.pt"
    if [ ! -f "$CKPT" ]; then
      echo "[에러] fold=${FOLD} seed=${SEED}: checkpoint 없음: $CKPT"
      exit 1
    fi
    if [ -z "$CKPTS" ]; then CKPTS="$CKPT"; else CKPTS="${CKPTS},${CKPT}"; fi
  done

  echo "=== model soup: fold=${FOLD} (3seed) Start: $(date) ==="
  python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed 42 \
      --backbone uni2 \
      --clinical-margin --clinical-staging --combine-mode cox_add \
      --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
      --fold "${FOLD}" --n-folds "${N_FOLDS}" \
      --eval-soup-ckpts "${CKPTS}"
  echo "=== model soup: fold=${FOLD} Complete: $(date) ==="
done
