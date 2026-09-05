#!/bin/bash
#SBATCH --job-name=PVT-recover-ext-pdaccons
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/recover_external_preds_pdaccons1500_%j.log

# 2026-09-05: train.py가 --fold/--n-folds(k-fold) 모드에서는 external_preds CSV를 저장하지
# 않는 gap을 발견(--full-train 모드에서만 저장하도록 짜여 있었음, train.py:3501 부근) — 이미
# 저장된 10개 체크포인트를 재학습 없이 다시 불러와 --eval-external-ckpt로 external 예측만
# 뽑아 CSV로 저장한다. GPU는 모델 forward(external 136명 평가)에만 쓰여 짧게 끝난다.
#
# 체크포인트 파일명을 하드코딩하지 않고 glob으로 자동 탐색 — 정확한 명명 규칙이 로컬 실험과
# 미묘하게 다를 수 있어(모델 prefix 조합 순서), seed+fold+레시피 핵심 태그로 유일하게 특정한다.
# 못 찾거나 여러 개 걸리면 후보를 출력하고 넘어간다(수동 확인 필요).
#
# 제출: sbatch sbatch/recover_external_preds_pdaccons1500_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5

for SEED in "${SEEDS[@]}"; do
  for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    PATTERN="models/checkpoint/survival_tcga_uni2native_seed${SEED}_*PDACCONS1500*CLR100_LRMW10_FOLD${FOLD}OF${N_FOLDS}_best_clinical_rna.pt"
    MATCHES=($(ls $PATTERN 2>/dev/null))
    if [ ${#MATCHES[@]} -eq 0 ]; then
      echo "!!! seed=${SEED} fold=${FOLD}: 체크포인트를 못 찾음 (패턴: ${PATTERN}) — 건너뜀"
      continue
    fi
    if [ ${#MATCHES[@]} -gt 1 ]; then
      echo "!!! seed=${SEED} fold=${FOLD}: 후보가 여러 개(${#MATCHES[@]}개) — 수동 확인 필요:"
      printf '    %s\n' "${MATCHES[@]}"
      continue
    fi
    CKPT="${MATCHES[0]}"
    echo "=== seed=${SEED} fold=${FOLD}/${N_FOLDS} ckpt=${CKPT} ==="
    python -u ./train.py --M4 --dataset tcga --external --seed "${SEED}" \
        --backbone uni2native --rna-genes pdac_consistency_1500 --use-cnv --clinical-mutation \
        --clinical-staging --clinical-margin --combine-mode cox_add \
        --clinical-lr-mult 100 --lr-mult-warmup-epochs 10 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" \
        --eval-external-ckpt "${CKPT}" 2>&1 | tee ".logs/recover_ext_pdaccons1500_seed${SEED}_fold${FOLD}.log"
  done
done

echo "=== 완료: .logs/external_preds/ 안에 10개 CSV가 생겼는지 확인할 것 ==="
ls -la .logs/external_preds/ | grep -i PDACCONS1500
