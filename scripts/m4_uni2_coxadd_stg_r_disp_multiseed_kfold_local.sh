#!/bin/bash
# M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD — PMA와 완전히 동일한 레시피(backbone/RNA gene set/
# clinical margin+staging/attn-dispersion/combine-mode cox_add/rna-aux-weight)에서 WSI pooling
# 구조만 바꾼 대조실험. PMA는 MultiComponentPooling(mean/std/attn/top-k 4관점)+CoAttentionPooling
# (RNA query)인 반면, M4는 단일 gated-ABMIL(models/vit_m1.py::AttentionPooling, RNA는 attention
# 게이트에 FiLM additive bias로만 개입) — 파라미터 훨씬 적은 WSI branch가 이 작은 코호트(~150명)
# 에서 덜 과적합해 더 잘 일반화하는지 확인한다(2026-08-14 논의: 지금까지 negative였던 실험은
# 전부 "데이터/설정" 레버였지 "모델 구조" 레버는 처음).
#
# train.py --M4 구성이 지금까지 --combine-mode/--clinical-margin/--attn-dispersion을 반영하지
# 않던 걸 이번에 --M4A와 동일하게 맞춰 고쳤다(train.py 2026-08-14).
#
# fold0/seed42 로컬 파일럿: internal test_c_index 0.4851(PMA)->0.6965(M4), external
# 0.5912(PMA)->0.5823(M4) — internal이 크게 뛰었지만 fold0/단일시드라 신뢰하지 않고 전체
# 스케일로 검증한다.
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 84 126)
N_FOLDS=5

for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_kfold5_fold${FOLD}.log"
    echo "=== M4(uni2,cox_add,STG+R,DISP,단일ABMIL) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
    python -u ./train.py --M4 --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --clinical-margin --clinical-staging --combine-mode cox_add \
        --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814m4_uni2_coxadd_stg_r_disp_multiseed_kfold5_local 2>&1 | tee "${log}"
    echo "=== M4(uni2,cox_add,STG+R,DISP,단일ABMIL) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
  done
done
