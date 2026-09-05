#!/bin/bash
#SBATCH --job-name=PVT-recover-ext-lit1500
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/recover_external_preds_literature1500_%j.log

# 2026-09-05: recover_external_preds_pdaccons1500_hpc.sh와 동일 목적/이유 — literature_1500
# 버전. model_prefix 로직상 "literature_1500"은 유전자 수를 태그에 안 남기고 그냥 "_EX"만
# 붙어(train.py 코드 확인 완료) 패턴이 짧고 겹칠 위험이 있다 — pdac_consistency_1500 잡과
# 같은 디렉토리에 있으니 "PDACCONS"가 없는 파일만 골라 구분한다.
#
# 제출: sbatch sbatch/recover_external_preds_literature1500_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5

for SEED in "${SEEDS[@]}"; do
  for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    PATTERN="models/checkpoint/survival_tcga_uni2native_seed${SEED}_*CLR100_LRMW10_FOLD${FOLD}OF${N_FOLDS}_best_clinical_rna.pt"
    MATCHES=($(ls $PATTERN 2>/dev/null | grep -v PDACCONS))
    if [ ${#MATCHES[@]} -eq 0 ]; then
      echo "!!! seed=${SEED} fold=${FOLD}: 체크포인트를 못 찾음 (패턴: ${PATTERN}, PDACCONS 제외) — 건너뜀"
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
        --backbone uni2native --rna-genes literature_1500 --use-cnv --clinical-mutation \
        --clinical-staging --clinical-margin --combine-mode cox_add \
        --clinical-lr-mult 100 --lr-mult-warmup-epochs 10 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" \
        --eval-external-ckpt "${CKPT}" 2>&1 | tee ".logs/recover_ext_literature1500_seed${SEED}_fold${FOLD}.log"
  done
done

echo "=== 완료: .logs/external_preds/ 안에 10개 CSV가 생겼는지 확인할 것 ==="
ls -la .logs/external_preds/ | grep -v PDACCONS
