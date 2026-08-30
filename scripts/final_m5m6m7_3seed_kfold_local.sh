#!/bin/bash
# 최종 확정용 M5(Clinical only)/M6(RNAseq only)/M7(Clinical+RNAseq, cox_add) 3seed(42/84/126)
# x 5fold 재현 — WSI가 없어 로컬에서 가볍게 돌린다(fold당 수십초~수분).
# 현재 코드베이스(stage-stratify 반영 이후) 기준 — paper/results_table_pma_family_3seed_kfold_ci.md의
# pre-stratify 점추정치(M5=0.564/0.532, M6=0.659/0.617, M7=0.637/0.622, CI 없음)를 대체할
# 최종 수치. train_light.py 사용(scripts/m5_m6_m7_multiseed_kfold_local.ps1과 동일 레시피).
#
# 로그는 .logs/가 아니라 paper/.log/에 남긴다(원본 예측 파일 덮어쓰기 사고 재발 방지 목적의
# 일환 — 단, kfold_preds/external_preds CSV 자체는 train_light.py에 하드코딩된 .logs/ 경로로
# 저장되는 건 동일하니, 완료 즉시 백업할 것).
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M5_STG_R_RAWLIN --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M6_INT1500 --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M7_INT1500_STG_R_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac --model M5_STG_R_RAWLIN --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac --model M6_INT1500 --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac --model M7_INT1500_STG_R_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 2026-08-21: M5는 raw_linear(ClinicalEncoder MLP 없이 raw feature -> Linear(1) 직결, 고전적
# Cox 회귀)를 최종으로 채택 — M2/M4/M7의 clinical cox_add도 전부 raw feature 직결이라(2026-08-21
# 원복), M5만 MLP를 쓸 architectural 근거가 없다는 사용자 지적(성능도 raw_linear가 오차범위 내로
# 동일/소폭 하락이라 통일하는 게 이득). 이제 clinical 브랜치는 전 모델에서 예외 없이 raw feature
# 직결 — RNA/WSI만 학습되는 인코더(RNAEncoder/ViT)를 쓴다는 일관된 아키텍처 원칙이 완성됨.
#
# 사용법: bash scripts/final_m5m6m7_3seed_kfold_local.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p paper/.log

SEEDS=(42 84 126)
N_FOLDS=5

declare -A MODEL_ARGS
MODEL_ARGS[M5]="--M5 --clinical-margin --clinical-staging --raw-linear"
MODEL_ARGS[M6]="--M6 --rna-genes literature_1500_intersection"
MODEL_ARGS[M7]="--M7 --rna-genes literature_1500_intersection --clinical-margin --clinical-staging --combine-mode cox_add"

for MODEL in M5 M6 M7; do
  for SEED in "${SEEDS[@]}"; do
    for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
      log="paper/.log/train_tcga_seed${SEED}_${MODEL}_kfold5_fold${FOLD}.log"
      echo "=== FINAL ${MODEL} seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
      python -u train_light.py ${MODEL_ARGS[$MODEL]} \
          --dataset tcga --external --seed "${SEED}" \
          --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0816_final_m5m6m7_3seed_kfold_local 2>&1 | tee "${log}"
      echo "=== FINAL ${MODEL} seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
    done
  done
done
echo "=== ALL FINAL M5/M6/M7 LOCAL 3SEED KFOLD RUNS COMPLETE: $(date) ==="
