#!/bin/bash
# 2026-09-09: M5/M6/M7(train_light.py)에 FINALEPOCH 저장을 새로 이식한 뒤 30개(2seed x 5fold x
# 3모델) 재학습 — M1/M2/M3/M4는 이미 FINALEPOCH 데이터가 있어 apples-to-apples 비교를 위해
# M5/M6/M7도 맞춘다(사용자 결정: "그래도 뭐 apple-to-apple은 맞춰야지. 재학습해도 다시 해줘").
cd "$(dirname "$0")/.."
set -o pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PathViT-ray

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

mkdir -p .logs
SEEDS=(84 126)
N_FOLDS=5
FAILED=()

run() {
    local log_file="$1"; shift
    echo "[RUN] $log_file  ($(date))"
    if "$@" 2>&1 | tee "$log_file"; then
        echo "=== Complete: $(date) ===" >> "$log_file"
    else
        echo "[FAIL] $log_file"
        FAILED+=("$log_file")
    fi
}

echo ">>> M5(ClinicalOnly, both-loss, FINALEPOCH) 재학습 시작: $(date)"
for SEED in "${SEEDS[@]}"; do
  for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    log=".logs/train_tcga_m5_final_fe_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"
    run "$log" python -u ./train_light.py --M5 --raw-linear --dataset tcga --external --seed "${SEED}" \
        --clinical-margin --clinical-staging \
        --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0909m5_final_fe_2seed_kfold5_local
  done
done

echo ">>> M6(RNAOnly, pdac_consistency_1500, both-loss, FINALEPOCH) 재학습 시작: $(date)"
for SEED in "${SEEDS[@]}"; do
  for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    log=".logs/train_tcga_m6_pdaccons_final_fe_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"
    run "$log" python -u ./train_light.py --M6 --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
        --use-cnv \
        --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0909m6_pdaccons_final_fe_2seed_kfold5_local
  done
done

echo ">>> M7(ClinicalRNAOnly, pdac_consistency_1500, both-loss, FINALEPOCH) 재학습 시작: $(date)"
for SEED in "${SEEDS[@]}"; do
  for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    log=".logs/train_tcga_m7_pdaccons_final_fe_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"
    run "$log" python -u ./train_light.py --M7 --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
        --use-cnv --clinical-margin --clinical-staging --clinical-mutation --combine-mode cox_add \
        --clinical-lr-mult 100 \
        --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0909m7_pdaccons_final_fe_2seed_kfold5_local
  done
done

echo "############################################################"
echo "# M5/M6/M7 FINALEPOCH 재학습 완료: $(date)  |  실패: ${#FAILED[@]}개"
if [ ${#FAILED[@]} -gt 0 ]; then
    for f in "${FAILED[@]}"; do echo "#   $f"; done
fi
echo "############################################################"
