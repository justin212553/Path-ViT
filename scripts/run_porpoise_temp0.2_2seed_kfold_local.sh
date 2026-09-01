#!/bin/bash
# 오늘 밤 sharpening 스윕(scripts/run_porpoise_sharpening_pilots_local.sh)에서 가장 좋았던
# temp=0.2(aux=0, seed84/fold0 단일 결과 C=0.7472, 오늘 나온 전체 PAAD 결과 중 최고)를
# 논문 최종 사양과 동일한 시드/폴드(2seed(84/126)x5fold, paper/results_table_pma_family_
# 3seed_kfold_ci.md 관례 — WSI 포함 모델은 seed42를 최종 집계에서 제외)로 검증한다.
# scripts/run_porpoise_no_aux_multiseed_kfold_local.sh와 동일 구조, seeds만 84/126 두 개로
# 줄이고 --porpoise-attn-temperature 0.2만 추가.
#
# HPC가 종일 점검 중이라 로컬 순차 실행.
#
# 사용법: PathViT-ray conda env에서
#   bash scripts/run_porpoise_temp0.2_2seed_kfold_local.sh
set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5

echo "=== [1/2] 학습: 2seed x 5fold = 10개 ==="
for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2_INT1500_SS_STG_R_T0.2_DISP_kfold5_fold${FOLD}.log"
    echo "--- seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ---"
    python -u ./train.py --PORPOISE --porpoise-attn-temperature 0.2 \
        --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --clinical-margin --clinical-staging \
        --patch-keep-frac 0.8 --attn-dispersion \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0901porpoise_temp0.2_2seed_kfold5_array 2>&1 | tee "${log}"
    echo "--- seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ---"
  done
done

echo "=== [2/2] external(cptac) eval-only: 10개 checkpoint 재사용 ==="
for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    # 2026-09-01: 예전엔 *STG_R_T0.2_DISP...로 느슨하게 찾다가, 어젯밤 aux=1.0 sharpening
    # 스윕이 남긴 "..._SS_AUX_STG_R_..._T0.2_DISP..." 체크포인트와 이번 aux=0 런의
    # "..._SS_STG_R_..._T0.2_DISP..."가 둘 다 매칭돼(seed84 fold0), 알파벳순으로 AUX 버전이
    # 먼저 골라져 rna_aux_head state_dict 불일치로 크래시했다 — "SS_STG_R"을 정확히 붙여서
    # AUX가 중간에 낀 버전을 구조적으로 배제한다.
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
echo "다음 명령으로 pooled 결과 확인:"
echo "  python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PORPOISE_uni2_INT1500_SS_STG_R_T0.2_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000"
echo "  python scripts/pool_multiseed_external_preds.py --dataset cptac --model PORPOISE_uni2_INT1500_SS_STG_R_T0.2_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000"
