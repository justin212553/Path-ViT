#!/bin/bash
# seed84/fold0 단일 fold 파일럿에서만 나왔던 "attention(no_aux) 0.7119 > meanpool 0.6784(-0.034)"가
# 진짜 아키텍처 효과인지, 이 프로젝트에서 반복돼온 "단일 fold 결과가 재현 안 되는" 패턴의
# 또 다른 사례인지 논문 사양(2seed(84/126)x5fold)으로 검증한다.
# scripts/run_porpoise_no_aux_multiseed_kfold_local.sh와 동일 구조, --porpoise-meanpool만 추가,
# seed는 84/126 두 개(paper 관례 — WSI 포함 모델은 seed42 제외)로 no_aux 베이스라인과 동일 조건.
#
# 사용법: PathViT-ray conda env에서
#   bash scripts/run_porpoise_meanpool_2seed_kfold_local.sh
set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5

echo "=== [1/2] 학습: 2seed x 5fold = 10개 ==="
for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2_INT1500_SS_STG_R_MEANPOOL_DISP_kfold5_fold${FOLD}.log"
    echo "--- seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ---"
    python -u ./train.py --PORPOISE --porpoise-meanpool --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --clinical-margin --clinical-staging \
        --patch-keep-frac 0.8 --attn-dispersion \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0901porpoise_meanpool_2seed_kfold5_array 2>&1 | tee "${log}"
    echo "--- seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ---"
  done
done

echo "=== [2/2] external(cptac) eval-only: 10개 checkpoint 재사용 ==="
for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    CKPT="models/checkpoint/survival_tcga_uni2_seed${SEED}_INT1500_SS_STG_R_PORPOISE_uni2_INT1500_SS_STG_R_MEANPOOL_DISP_FOLD${FOLD}OF${N_FOLDS}_best_porpoise.pt"
    if [ ! -e "${CKPT}" ]; then
      echo "[SKIP] seed=${SEED} fold=${FOLD}: checkpoint를 못 찾음 (${CKPT})"
      continue
    fi
    echo "--- external eval-only: seed=${SEED} fold=${FOLD} ckpt=${CKPT} Start: $(date) ---"
    python -u ./train.py --PORPOISE --porpoise-meanpool --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --clinical-margin --clinical-staging \
        --patch-keep-frac 0.8 --attn-dispersion \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" \
        --eval-external-ckpt "${CKPT}"
    echo "--- external eval-only: seed=${SEED} fold=${FOLD} Complete: $(date) ---"
  done
done

echo "=== 전부 완료: $(date) ==="
echo "다음 명령으로 pooled 결과 확인:"
echo "  python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PORPOISE_uni2_INT1500_SS_STG_R_MEANPOOL_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000"
echo "  python scripts/pool_multiseed_external_preds.py --dataset cptac --model PORPOISE_uni2_INT1500_SS_STG_R_MEANPOOL_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000"
