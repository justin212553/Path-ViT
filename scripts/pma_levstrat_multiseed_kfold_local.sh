#!/bin/bash
# PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_LEVSTRAT — stage-stratify와 동일한 조사(fold별
# internal log-rank p 변동 원인)의 후속 실험. 다변량 상관 분석에서 고레버리지 환자 집중도
# (rho=0.894, data/dataset.py::_HIGH_LEVERAGE_CASE_IDS)가 stage 다음으로 강한 후보였다 —
# 이걸 split stratification key에 직접 추가해 fold 간 쏠림을 통제한다(--stage-stratify는 끔,
# 단일 요인 비교를 위해). 로컬(HPC 아님)에서 3seed(42/84/126) x 5fold 전체 검증.
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 84 126)
N_FOLDS=5

for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_LEVSTRAT_kfold5_fold${FOLD}.log"
    echo "=== PMA(uni2,cox_add,STG+R,leverage-stratify) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
    python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --clinical-margin --clinical-staging --combine-mode cox_add \
        --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
        --leverage-stratify \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814_pma_levstrat_multiseed_kfold5_local 2>&1 | tee "${log}"
    echo "=== PMA(uni2,cox_add,STG+R,leverage-stratify) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
  done
done
