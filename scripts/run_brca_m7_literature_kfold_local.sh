#!/bin/bash
# 2026-09-03: scripts/run_brca_m7_coxfdr_stg_kfold_local.sh와 정확히 같은 조건(--clinical-staging
# --external-tss none, seed 84/126 x fold 0..4)으로 문헌 기반 패널(PAM50+Oncotype DX 60유전자,
# scripts/select_brca_rna_genes_literature.py)을 돌린다 — cox-FDR 패널과 나란히 비교용.
# PAAD에서 pathway8(문헌 큐레이션, 생존 라벨 미사용)이 Cox+FDR보다 나았던 패턴이 BRCA에서도
# 재현되는지 확인.
#
# 실행: bash scripts/run_brca_m7_literature_kfold_local.sh > .logs/brca_m7_literature_kfold.log 2>&1

set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5
COMMON="--gene-selection literature --clinical-staging --external-tss none --epochs 100 --patience 20 --group-ts 0903_brca_m7_literature_kfold_local"

for SEED in "${SEEDS[@]}"; do
  for FOLD in 0 1 2 3 4; do
    echo "=== M7 literature+stg seed=${SEED} fold=${FOLD}/${N_FOLDS} $(date) ==="
    python -u -m scripts.train_brca_m7 ${COMMON} --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" 2>&1 | tail -8
  done
done

echo
echo "=== ALL DONE $(date) ==="
echo "--- pooled internal (literature panel) ---"
python scripts/pool_multiseed_kfold_preds.py --dataset brca --model BRCA_M7_LIT60_STG --seeds 84,126 --n-folds 5 --bootstrap 2000
