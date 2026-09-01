#!/bin/bash
# sbatch/porpoise_no_aux_multiseed_kfold_array_hpc.sh + sbatch/porpoise_no_aux_multiseed_external_eval_hpc.sh
# 를 SLURM array 대신 로컬에서 순차 실행하는 버전 — HPC 점검 중일 때 로컬 GPU 리소스로
# 같은 검증(3seed x 5fold 학습 -> external eval-only)을 대신 돌린다. 로컬에 tcga/cptac
# uni2 feature가 전부 있어(data/patches_{tcga,cptac}/tiles/*/features_uni2.pt) 재현 가능.
#
# 사용법: PathViT-ray conda env에서
#   bash scripts/run_porpoise_no_aux_multiseed_kfold_local.sh
set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEEDS=(42 84 126)
N_FOLDS=5

echo "=== [1/2] 학습: 3seed x 5fold = 15개 ==="
for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2_INT1500_SS_STG_R_DISP_kfold5_fold${FOLD}.log"
    echo "--- seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ---"
    python -u ./train.py --PORPOISE --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --clinical-margin --clinical-staging \
        --patch-keep-frac 0.8 --attn-dispersion \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0831porpoise_no_aux_multiseed_kfold5_array 2>&1 | tee "${log}"
    echo "--- seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ---"
  done
done

echo "=== [2/2] external(cptac) eval-only: 15개 checkpoint 재사용 ==="
for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    MATCHES=(models/checkpoint/survival_tcga_uni2_seed${SEED}_*STG_R_DISP_FOLD${FOLD}OF${N_FOLDS}_best_porpoise.pt)
    if [ ! -e "${MATCHES[0]}" ]; then
      echo "[SKIP] seed=${SEED} fold=${FOLD}: checkpoint를 못 찾음"
      continue
    fi
    if [ "${#MATCHES[@]}" -gt 1 ]; then
      echo "[경고] seed=${SEED} fold=${FOLD}: checkpoint가 ${#MATCHES[@]}개 매칭됨 — 첫 번째만 사용: ${MATCHES[0]}"
    fi
    CKPT="${MATCHES[0]}"
    echo "--- external eval-only: seed=${SEED} fold=${FOLD} ckpt=${CKPT} Start: $(date) ---"
    python -u ./train.py --PORPOISE --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
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
echo "  python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PORPOISE_uni2_INT1500_SS_STG_R_DISP --seeds 42,84,126 --n-folds 5 --bootstrap 2000"
echo "  python scripts/pool_multiseed_external_preds.py --dataset cptac --model PORPOISE_uni2_INT1500_SS_STG_R_DISP --seeds 42,84,126 --n-folds 5 --bootstrap 2000"
