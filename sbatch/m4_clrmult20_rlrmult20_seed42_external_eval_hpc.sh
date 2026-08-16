#!/bin/bash
#SBATCH --job-name=PVT-M4-clrlr20-rlr20-seed42-ext-eval
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m4_clrmult20_rlrmult20_seed42_external_eval.log

# m4_clrmult20_rlrmult20_seed42_kfold_array_hpc.sh(seed42, 5fold 학습)가 저장해 둔 checkpoint
# 5개를 재학습 없이 다시 불러와 --eval-external-ckpt로 external(cptac) 예측만 재추출한다.
#
# 완료 후(.logs/external_preds/에 CSV 5개 있는지 확인):
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT_CLR20_RLR20 --seeds 42 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m4_clrmult20_rlrmult20_seed42_external_eval_hpc.sh
# (5개 학습 job이 전부 끝나 checkpoint가 다 저장된 뒤에 제출할 것)

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=42
N_FOLDS=5

for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
  mapfile -t MATCHES < <(ls models/checkpoint/survival_tcga_uni2_seed${SEED}_*M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT_CLR20_RLR20_FOLD${FOLD}OF${N_FOLDS}_best_clinical_rna.pt 2>/dev/null)
  if [ "${#MATCHES[@]}" -eq 0 ]; then
    echo "[SKIP] fold=${FOLD}: checkpoint를 못 찾음 (학습이 아직 안 끝났거나 경로가 다름)"
    continue
  fi
  if [ "${#MATCHES[@]}" -gt 1 ]; then
    echo "[경고] fold=${FOLD}: checkpoint가 ${#MATCHES[@]}개 매칭됨 — 첫 번째만 사용: ${MATCHES[0]}"
  fi
  CKPT="${MATCHES[0]}"

  echo "=== external eval-only: fold=${FOLD} ckpt=${CKPT} Start: $(date) ==="
  python -u ./train.py --M4 --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
      --backbone uni2 \
      --clinical-margin --clinical-staging --combine-mode cox_add \
      --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
      --skip-patch-vit \
      --clinical-lr-mult 20.0 --rna-lr-mult 20.0 \
      --fold "${FOLD}" --n-folds "${N_FOLDS}" \
      --eval-external-ckpt "${CKPT}"
  echo "=== external eval-only: fold=${FOLD} Complete: $(date) ==="
done
