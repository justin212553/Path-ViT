#!/bin/bash
# 2026-09-04: sbatch/brca_m4_consistency_stg_kfold_array_hpc.sh(HPC, M4)와 정확히 같은 조건
# (--gene-selection consistency --clinical-staging --external-tss none,
# seed 84/126 x fold 0..4)으로 M7을 로컬에서 돌린다 — M7은 WSI가 없어 빨라서 HPC 자리를
# 안 쓰고도 M4와 paired bootstrap 비교할 짝을 맞출 수 있다.
#
# 실행: bash scripts/run_brca_m7_consistency_stg_kfold_local.sh > .logs/brca_m7_consistency_stg_kfold.log 2>&1

set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5
COMMON="--gene-selection consistency --clinical-staging --external-tss none --epochs 100 --patience 20 --group-ts 0904_brca_m7_consistency_stg_kfold_local"

for SEED in "${SEEDS[@]}"; do
  for FOLD in 0 1 2 3 4; do
    echo "=== M7 consistency+stg seed=${SEED} fold=${FOLD}/${N_FOLDS} $(date) ==="
    python -u -m scripts.train_brca_m7 ${COMMON} --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" 2>&1 | tail -8
  done
done

echo
echo "=== ALL DONE $(date) ==="
echo "pooled 비교(M4 HPC 완료 후):"
echo "  python scripts/pool_multiseed_kfold_preds.py --dataset brca --model BRCA_M7_CONS882_STG --seeds 84,126 --n-folds 5 --bootstrap 2000"
echo "  python scripts/pool_multiseed_kfold_preds.py --dataset brca --model BRCA_PMA_CONS882_STG_SS_AUX --seeds 84,126 --n-folds 5 --bootstrap 2000"
echo "  python scripts/paired_bootstrap_delta.py --dataset brca --model-a BRCA_M7_CONS882_STG --model-b BRCA_PMA_CONS882_STG_SS_AUX --seeds 84,126 --n-folds 5 --bootstrap 2000"
