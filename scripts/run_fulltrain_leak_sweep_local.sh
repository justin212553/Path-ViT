#!/bin/bash
# 2026-09-02: 사용자 제안 "approach #1" — TCGA 전체를 --full-train으로 학습(내부 split 없음,
# 고정 epoch), external(CPTAC)만 보고. internal k-fold pooled c-index가 fold당 표본이 작고
# (N~30) cross-fold 모델 보정 불일치까지 겹쳐 신뢰하기 어렵다는 오늘의 발견(findings_backlog.md)
# 때문에, 시드만 여러 개 반복해 external 평균±CI로 판단하는 대안 프로토콜.
#
# 6개 설정 x 5개 시드 = 30 run, 각 ~70초(로컬, WSI 없음) -> 총 35분 내외 예상.
# 1) M7 literature_1500(leak 있음) baseline
# 2) M7 pathway8(leak 없음) baseline
# 3-6) M7 pathway8 + clinical-lr-mult 5/10/20/50
#
# 실행: bash scripts/run_fulltrain_leak_sweep_local.sh > .logs/fulltrain_leak_sweep_summary.log 2>&1

set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEEDS=(42 84 126 168 210)
EPOCHS=30
COMMON="--M7 --dataset tcga --external --clinical-margin --clinical-staging --combine-mode cox_add --full-train --epochs ${EPOCHS}"

declare -A CONFIGS
CONFIGS[1_INT1500_baseline]="--rna-genes literature_1500_intersection"
CONFIGS[2_PW8_baseline]="--rna-genes pathway8"
CONFIGS[3_PW8_CLR5]="--rna-genes pathway8 --clinical-lr-mult 5"
CONFIGS[4_PW8_CLR10]="--rna-genes pathway8 --clinical-lr-mult 10"
CONFIGS[5_PW8_CLR20]="--rna-genes pathway8 --clinical-lr-mult 20"
CONFIGS[6_PW8_CLR50]="--rna-genes pathway8 --clinical-lr-mult 50"

for KEY in "${!CONFIGS[@]}"; do
  ARGS="${CONFIGS[$KEY]}"
  echo "########## CONFIG ${KEY}: ${ARGS} ##########"
  for SEED in "${SEEDS[@]}"; do
    echo "=== ${KEY} seed=${SEED} $(date) ==="
    python -u train_light.py ${COMMON} ${ARGS} --seed "${SEED}" --group-ts fulltrain_leak_sweep_0902 2>&1 | tail -6
  done
done

echo
echo "##################################################"
echo "########## SUMMARY (bootstrap 95% CI) ###########"
echo "##################################################"

SEED_LIST="42,84,126,168,210"
echo
echo "--- 1) M7 literature_1500 (leak) baseline ---"
python scripts/pool_fulltrain_external_preds.py --dataset cptac --model M7_INT1500_STG_R_COX_ADD --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- 2) M7 pathway8 (no-leak) baseline ---"
python scripts/pool_fulltrain_external_preds.py --dataset cptac --model M7_PW8_STG_R_COX_ADD --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- 3) M7 pathway8 + clinical-lr-mult=5 ---"
python scripts/pool_fulltrain_external_preds.py --dataset cptac --model M7_PW8_STG_R_COX_ADD_CLR5 --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- 4) M7 pathway8 + clinical-lr-mult=10 ---"
python scripts/pool_fulltrain_external_preds.py --dataset cptac --model M7_PW8_STG_R_COX_ADD_CLR10 --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- 5) M7 pathway8 + clinical-lr-mult=20 ---"
python scripts/pool_fulltrain_external_preds.py --dataset cptac --model M7_PW8_STG_R_COX_ADD_CLR20 --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "--- 6) M7 pathway8 + clinical-lr-mult=50 ---"
python scripts/pool_fulltrain_external_preds.py --dataset cptac --model M7_PW8_STG_R_COX_ADD_CLR50 --seeds "${SEED_LIST}" --bootstrap 2000

echo
echo "=== ALL DONE $(date) ==="
