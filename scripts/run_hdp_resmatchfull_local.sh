#!/bin/bash
# 2026-09-02: HDP_Pretrain_Cluster를 해상도 보정 head(val_corr 0.60->0.78) + 전체 tile 트리
# 커버리지(pretrain_resmatch_full, TCGA 152/152 완전 커버) 조합으로 재검증 — 기존 검증판
# (HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8[_CLR10], paired bootstrap p=0.132/0.362, 이 세션
# 최고 기록)과 비교. 원래 HPC에서 도는 게 정상이지만 자리가 안 나서 이제 로컬(uni2native
# tile 트리를 zip으로 받아옴)에서 대신 돈다.
#
# 2개 설정 x 5시드(42/84/126/168/210) = 10 run. WSI forward가 있어 PMA와 비슷하게 run당
# 수 분 예상.
#
# 실행: bash scripts/run_hdp_resmatchfull_local.sh > .logs/hdp_resmatchfull_summary.log 2>&1

set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEEDS=(42 84 126 168 210)
HEAD="data/hdp_pretrain_tumor_content_head_resmatch.pt"
COMMON="--dataset tcga --external --rna-genes pathway8 --head-path ${HEAD} --pretrain-source pretrain_resmatch_full --full-train --epochs 30"

declare -A CONFIGS
CONFIGS[1_baseline]=""
CONFIGS[2_CLR10]="--clinical-lr-mult 10"

for KEY in "${!CONFIGS[@]}"; do
  ARGS="${CONFIGS[$KEY]}"
  echo "########## CONFIG ${KEY}: ${ARGS} ##########"
  for SEED in "${SEEDS[@]}"; do
    echo "=== ${KEY} seed=${SEED} $(date) ==="
    python -u train_hdp_pretrain_cluster.py ${COMMON} ${ARGS} --seed "${SEED}" \
        --group-ts hdp_resmatchfull_local_0902 2>&1 | tail -6
  done
done

echo
echo "##################################################"
echo "########## SUMMARY (bootstrap 95% CI) ###########"
echo "##################################################"

SEED_LIST="42,84,126,168,210"
echo
echo "--- 1) HDP RESMATCHFULL baseline ---"
python scripts/pool_fulltrain_external_preds.py --dataset cptac --model HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8_RESMATCHFULL --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- 2) HDP RESMATCHFULL + clinical-lr-mult=10 ---"
python scripts/pool_fulltrain_external_preds.py --dataset cptac --model HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8_RESMATCHFULL_CLR10 --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "##################################################"
echo "##### PAIRED BOOTSTRAP vs M7(RNA+Clinical만) #####"
echo "##################################################"

echo
echo "--- M7(PW8) vs HDP RESMATCHFULL baseline ---"
python scripts/paired_bootstrap_delta_fulltrain.py --dataset cptac \
    --model-a M7_PW8_STG_R_COX_ADD --model-b HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8_RESMATCHFULL \
    --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- M7(PW8+CLR10) vs HDP RESMATCHFULL+CLR10 ---"
python scripts/paired_bootstrap_delta_fulltrain.py --dataset cptac \
    --model-a M7_PW8_STG_R_COX_ADD_CLR10 --model-b HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8_RESMATCHFULL_CLR10 \
    --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- (참고) 기존 검증판(h5, non-resmatch) vs RESMATCHFULL 직접 비교 ---"
python scripts/paired_bootstrap_delta_fulltrain.py --dataset cptac \
    --model-a HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8 --model-b HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8_RESMATCHFULL \
    --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "=== ALL DONE $(date) ==="
