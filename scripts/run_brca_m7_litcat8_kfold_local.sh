#!/bin/bash
# 2026-09-03: sbatch/brca_m4_litcat8_stg_kfold_array_hpc.sh(HPC, M4)와 정확히 같은 조건
# (--gene-selection literature_categorized --clinical-staging --external-tss none,
# seed 84/126 x fold 0..4)으로 M7을 로컬에서 돌린다.
#
# 실행: bash scripts/run_brca_m7_litcat8_kfold_local.sh > .logs/brca_m7_litcat8_kfold.log 2>&1

set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5
COMMON="--gene-selection literature_categorized --clinical-staging --external-tss none --epochs 100 --patience 20 --group-ts 0903_brca_m7_litcat8_kfold_local"

for SEED in "${SEEDS[@]}"; do
  for FOLD in 0 1 2 3 4; do
    echo "=== M7 litcat8+stg seed=${SEED} fold=${FOLD}/${N_FOLDS} $(date) ==="
    python -u -m scripts.train_brca_m7 ${COMMON} --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" 2>&1 | tail -8
  done
done

echo
echo "=== ALL DONE $(date) ==="
python scripts/pool_multiseed_kfold_preds.py --dataset brca --model BRCA_M7_LITCAT8_STG --seeds 84,126 --n-folds 5 --bootstrap 2000
