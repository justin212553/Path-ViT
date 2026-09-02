#!/bin/bash
# 2026-09-02: M7(RNA+Clinical만)에 이어 PMA(M4, WSI+RNA+Clinical)를 같은 프로토콜로 검증 —
# TCGA 전체 --full-train, 고정 30epoch, external(CPTAC)만, 5시드(42/84/126/168/210) 반복.
# 목적: (1) literature_1500 vs pathway8 gene panel 비교를 WSI 포함 모델에서도 재확인,
# (2) clinical-lr-mult=10(어젯밤 M7에서 최고점, 사용자 지시로 스윕 대신 10 하나만) 효과 확인,
# (3) 이후 scripts/paired_bootstrap_delta_fulltrain.py로 M7(RNA+Clinical) 대비 WSI 추가가
# external에서 유의한 순증분인지 검정할 pooled prediction CSV 생성.
#
# 4개 설정 x 5시드 = 20 run, PMA는 WSI forward가 있어 run당 ~5분(로컬, backbone=uni2 캐시됨) ->
# 총 ~100분 예상.
#
# 실행: bash scripts/run_pma_leak_sweep_local.sh > .logs/pma_leak_sweep_summary.log 2>&1

set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEEDS=(42 84 126 168 210)
EPOCHS=30
COMMON="--PMA --dataset tcga --external --backbone uni2 --clinical-margin --clinical-staging --combine-mode cox_add --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 --full-train --epochs ${EPOCHS}"

declare -A CONFIGS
CONFIGS[1_PMA_INT1500_baseline]="--rna-genes literature_1500_intersection"
CONFIGS[2_PMA_INT1500_CLR10]="--rna-genes literature_1500_intersection --clinical-lr-mult 10"
CONFIGS[3_PMA_PW8_baseline]="--rna-genes pathway8"
CONFIGS[4_PMA_PW8_CLR10]="--rna-genes pathway8 --clinical-lr-mult 10"

for KEY in "${!CONFIGS[@]}"; do
  ARGS="${CONFIGS[$KEY]}"
  echo "########## CONFIG ${KEY}: ${ARGS} ##########"
  for SEED in "${SEEDS[@]}"; do
    echo "=== ${KEY} seed=${SEED} $(date) ==="
    python -u train.py ${COMMON} ${ARGS} --seed "${SEED}" --group-ts pma_leak_sweep_0902 2>&1 | tail -6
  done
done

echo
echo "##################################################"
echo "########## SUMMARY (bootstrap 95% CI) ###########"
echo "##################################################"

SEED_LIST="42,84,126,168,210"
echo
echo "--- 1) PMA literature_1500 baseline ---"
python scripts/pool_fulltrain_external_preds.py --dataset cptac --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- 2) PMA literature_1500 + clinical-lr-mult=10 ---"
python scripts/pool_fulltrain_external_preds.py --dataset cptac --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_CLR10 --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- 3) PMA pathway8 baseline ---"
python scripts/pool_fulltrain_external_preds.py --dataset cptac --model PMA_uni2_PW8_SS_AUX_STG_R_DISP_COX_ADD --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- 4) PMA pathway8 + clinical-lr-mult=10 ---"
python scripts/pool_fulltrain_external_preds.py --dataset cptac --model PMA_uni2_PW8_SS_AUX_STG_R_DISP_COX_ADD_CLR10 --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "##################################################"
echo "##### PAIRED BOOTSTRAP vs M7(RNA+Clinical만) #####"
echo "##################################################"

echo
echo "--- M7(INT1500) vs PMA(INT1500): WSI 추가 유의성 ---"
python scripts/paired_bootstrap_delta_fulltrain.py --dataset cptac \
    --model-a M7_INT1500_STG_R_COX_ADD --model-b PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD \
    --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- M7(PW8) vs PMA(PW8): WSI 추가 유의성(leak 없는 RNA 기준) ---"
python scripts/paired_bootstrap_delta_fulltrain.py --dataset cptac \
    --model-a M7_PW8_STG_R_COX_ADD --model-b PMA_uni2_PW8_SS_AUX_STG_R_DISP_COX_ADD \
    --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- M7(PW8+CLR10) vs PMA(PW8+CLR10): WSI 추가 유의성(clinical 살린 기준) ---"
python scripts/paired_bootstrap_delta_fulltrain.py --dataset cptac \
    --model-a M7_PW8_STG_R_COX_ADD_CLR10 --model-b PMA_uni2_PW8_SS_AUX_STG_R_DISP_COX_ADD_CLR10 \
    --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "=== ALL DONE $(date) ==="
