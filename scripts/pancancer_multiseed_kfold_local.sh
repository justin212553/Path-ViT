#!/bin/bash
# PAAD+BRCA 공동학습(WSI trunk weight-tying, UNI v1) 3seed(42/84/126) x 5fold 로컬 순차 실행.
# 단일 seed=84 파일럿(internal 0.6911 best-ckpt / 0.7602 final, external 0.6407/0.6513 —
# baseline 대비 뚜렷한 개선)이 재현되는지 확인한다. 1회 ~6분이라 15회 순차로 로컬에서
# 감당 가능(HPC 불필요).
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PANCANCER_PAAD_BRCA_uni_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PANCANCER_PAAD_BRCA_uni_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 84 126)
N_FOLDS=5

for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_pancancer_paad_brca_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"
    echo "=== pancancer PAAD+BRCA seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
    python -u -m scripts.train_pancancer_paad_brca --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
        --group-ts 0811pancancer_paad_brca_multiseed 2>&1 | tee "${log}"
    echo "=== pancancer PAAD+BRCA seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
  done
done
