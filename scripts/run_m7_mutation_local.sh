#!/bin/bash
# 2026-09-02: M7(pathway8) + KRAS/TP53/SMAD4/CDKN2A mutation flags(clinical에 얹음) — mutation이
# margin/staging 위에 순증분을 주는지 검증. seed84는 이미 스모크 테스트로 30epoch 다 돌았으니
# 재사용(스크립트가 같은 이름으로 덮어써도 값은 사실상 같음 - 결정적이지 않은 미세한 GPU 차이만).
#
# 실행: bash scripts/run_m7_mutation_local.sh > .logs/m7_mutation_summary.log 2>&1

set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEEDS=(42 84 126 168 210)
COMMON="--M7 --dataset tcga --external --rna-genes pathway8 --clinical-margin --clinical-staging --clinical-mutation --combine-mode cox_add --full-train --epochs 30"

for SEED in "${SEEDS[@]}"; do
  echo "=== M7 pathway8+mutation seed=${SEED} $(date) ==="
  python -u train_light.py ${COMMON} --seed "${SEED}" --group-ts m7_mutation_local_0902 2>&1 | tail -6
done

echo
echo "##################################################"
echo "########## SUMMARY (bootstrap 95% CI) ###########"
echo "##################################################"

SEED_LIST="42,84,126,168,210"
echo
echo "--- M7 pathway8 + mutation ---"
python scripts/pool_fulltrain_external_preds.py --dataset cptac --model M7_PW8_STG_R_MUT_COX_ADD --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- M7 pathway8(mutation 없음, 어젯밤 baseline) vs +mutation: 순증분 유의성 ---"
python scripts/paired_bootstrap_delta_fulltrain.py --dataset cptac \
    --model-a M7_PW8_STG_R_COX_ADD --model-b M7_PW8_STG_R_MUT_COX_ADD \
    --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "=== ALL DONE $(date) ==="
