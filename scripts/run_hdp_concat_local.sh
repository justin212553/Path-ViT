#!/bin/bash
# 2026-09-02: combine_mode="concat"(clinical/hist/growth/maturity에 RNA와 동등한 nonlinear
# encoder를 주고, 스칼라 조기압축 대신 concat 후 hidden layer 있는 공유 risk_head로 합침)
# 검증 — diagnose_hdp_checkpoint_weights.py가 cox_add에서 RNA 99.7% 독식/WSI 전부 ~0%를
# 보인 뒤, "branch 표현력 비대칭이 원인"이라는 가설을 아키텍처로 직접 검증한다.
#
# pathway8(leak 없음) baseline만, 5시드(42/84/126/168/210). CLR10은 이 축과 섞으면
# 해석이 꼬여서(둘 다 branch-competition 개입) 이번엔 뺀다.
#
# 실행: bash scripts/run_hdp_concat_local.sh > .logs/hdp_concat_summary.log 2>&1

set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEEDS=(42 84 126 168 210)
COMMON="--dataset tcga --external --rna-genes pathway8 --combine-mode concat --full-train --epochs 30"

for SEED in "${SEEDS[@]}"; do
  echo "=== concat baseline seed=${SEED} $(date) ==="
  python -u train_hdp_pretrain_cluster.py ${COMMON} --seed "${SEED}" \
      --group-ts hdp_concat_local_0902 2>&1 | tail -6
done

echo
echo "##################################################"
echo "########## SUMMARY (bootstrap 95% CI) ###########"
echo "##################################################"

SEED_LIST="42,84,126,168,210"
echo
echo "--- HDP concat baseline ---"
python scripts/pool_fulltrain_external_preds.py --dataset cptac --model HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8_CONCAT --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- M7(PW8) vs HDP concat: WSI 추가 유의성 ---"
python scripts/paired_bootstrap_delta_fulltrain.py --dataset cptac \
    --model-a M7_PW8_STG_R_COX_ADD --model-b HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8_CONCAT \
    --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- cox_add(동일 head/coverage 기준) vs concat: 아키텍처 자체의 효과(다른 축 안 섞임) ---"
python scripts/paired_bootstrap_delta_fulltrain.py --dataset cptac \
    --model-a HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8 --model-b HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8_CONCAT \
    --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "=== ALL DONE $(date) ==="
