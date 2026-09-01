#!/bin/bash
# scripts/run_porpoise_temp0.2_2seed_kfold_local.sh의 [2/2] external eval 단계만 따로 재실행
# — 학습(phase 1)은 이미 10개 다 성공했는데, 원래 스크립트의 체크포인트 glob이 너무 느슨해서
# (*STG_R_T0.2_DISP...) 어젯밤 aux=1.0 sharpening 스윕이 남긴 "..._SS_AUX_STG_R_..." 체크포인트
# (seed84 fold0)와 충돌 — 알파벳순으로 AUX 버전이 먼저 골라져 rna_aux_head state_dict 불일치로
# 크래시했다(2026-09-01). glob을 "SS_STG_R"로 정확히 고쳐 재실행.
set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5

echo "=== external(cptac) eval-only: 10개 checkpoint 재사용 ==="
for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    MATCHES=(models/checkpoint/survival_tcga_uni2_seed${SEED}_INT1500_SS_STG_R_PORPOISE_uni2_INT1500_SS_STG_R_T0.2_DISP_FOLD${FOLD}OF${N_FOLDS}_best_porpoise.pt)
    if [ ! -e "${MATCHES[0]}" ]; then
      echo "[SKIP] seed=${SEED} fold=${FOLD}: checkpoint를 못 찾음"
      continue
    fi
    if [ "${#MATCHES[@]}" -gt 1 ]; then
      echo "[경고] seed=${SEED} fold=${FOLD}: checkpoint가 ${#MATCHES[@]}개 매칭됨 — 첫 번째만 사용: ${MATCHES[0]}"
    fi
    CKPT="${MATCHES[0]}"
    echo "--- external eval-only: seed=${SEED} fold=${FOLD} ckpt=${CKPT} Start: $(date) ---"
    python -u ./train.py --PORPOISE --porpoise-attn-temperature 0.2 \
        --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --clinical-margin --clinical-staging \
        --patch-keep-frac 0.8 --attn-dispersion \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" \
        --eval-external-ckpt "${CKPT}"
    echo "--- external eval-only: seed=${SEED} fold=${FOLD} Complete: $(date) ---"
  done
done

echo "=== 전부 완료: $(date) ==="
