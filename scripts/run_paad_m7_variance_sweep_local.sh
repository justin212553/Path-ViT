#!/bin/bash
# 2026-09-03: PAAD M7 variance 기반(생존 라벨 미사용) single-cohort RNA 패널 유전자 수 스윕.
# BRCA에서 확인된 것 — Cox 선택(라벨 직접 사용)은 fold 경계 leak에 취약했지만 variance
# 선택(라벨 미사용)은 훨씬 덜 취약했다(findings_backlog.md 2026-09-03) — 이걸 PAAD에도
# 적용해보되, PAAD(TCGA 152명)는 BRCA(1057명)보다 훨씬 작아 1500개를 그대로 쓰는 게 무리일
# 수 있다는 판단 하에 100/250/500/1000/1500 단계별로 비교한다(사용자 지시, "코호트 자체가
# 작으니 유전자 수를 1500개 그대로 가져가는 건 무리가 있을 것 같고").
#
# 기존 leak 정량화/재검증과 동일한 레시피(--clinical-staging --clinical-margin --combine-mode
# cox_add, 2seed(84/126)x5fold)를 그대로 쓴다 — pathway8(0.557/0.604)·intersection(0.655/0.622)·
# Cox+FDR(0.686/0.595)와 나란히 비교 가능하게.
#
# [k-fold external CSV 저장 안 되는 문제] train_light.py는 --fold(k-fold) 모드에서 external
# 평가는 하지만 CSV 저장은 --full-train 때만 한다(2026-09-02 확인) — 그래서 학습 뒤에
# --eval-external-ckpt로 저장된 체크포인트를 다시 읽어 external CSV를 만드는 2단계 구조로 짠다.
#
# 실행(오래 걸림, 백그라운드 권장):
#   bash scripts/run_paad_m7_variance_sweep_local.sh > .logs/paad_m7_variance_sweep.log 2>&1

set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

N_GENES_LIST=(100 250 500 1000 1500)
SEEDS=(84 126)
N_FOLDS=5

for N in "${N_GENES_LIST[@]}"; do
  RNA_GENES="variance_${N}_tcga_only"
  MODEL_BASE="M7_EXTVAR${N}_STG_R_COX_ADD"
  COMMON="--M7 --dataset tcga --external --rna-genes ${RNA_GENES} --clinical-staging --clinical-margin --combine-mode cox_add --group-ts 0903_paad_m7_variance_sweep_local"

  echo "############################################################"
  echo "########## n_genes=${N} 학습 시작 $(date) ##########"
  echo "############################################################"
  for SEED in "${SEEDS[@]}"; do
    for FOLD in 0 1 2 3 4; do
      echo "=== n_genes=${N} seed=${SEED} fold=${FOLD}/${N_FOLDS} $(date) ==="
      python -u train_light.py ${COMMON} --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" 2>&1 | tail -6
    done
  done

  echo "--- n_genes=${N}: external CSV 재생성(eval-external-ckpt) ---"
  for SEED in "${SEEDS[@]}"; do
    for FOLD in 0 1 2 3 4; do
      CKPT="models/checkpoint/survival_tcga_best_$(echo ${MODEL_BASE}_FOLD${FOLD}OF${N_FOLDS} | tr '[:upper:]' '[:lower:]')_seed${SEED}_light.pt"
      python -u train_light.py ${COMMON} --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" --eval-external-ckpt "${CKPT}" 2>&1 | tail -4
    done
  done

  echo "--- n_genes=${N}: pooled internal/external ---"
  python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model "${MODEL_BASE}" --seeds 84,126 --n-folds 5 --bootstrap 2000
  python scripts/pool_multiseed_external_preds.py --dataset cptac --model "${MODEL_BASE}" --seeds 84,126 --n-folds 5 --bootstrap 2000
done

echo
echo "=== ALL DONE $(date) ==="
