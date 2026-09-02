#!/bin/bash
#SBATCH --job-name=PVT-HDPPC-ext-eval
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/hdp_pretrain_cluster_multiseed_external_eval.log

# hdp_pretrain_cluster_multiseed_kfold_array_hpc.sh(2seed x 5fold=10개 학습)가 저장해 둔
# checkpoint 10개를 재학습 없이 다시 불러와 external(cptac) 예측만 재추출한다.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model HDP_PRETRAIN_CLUSTER_INT1500_STG_R_GROWTH8 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model HDP_PRETRAIN_CLUSTER_INT1500_STG_R_GROWTH8 --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출(10개 학습 job이 전부 끝나 checkpoint가 다 저장된 뒤에): sbatch sbatch/hdp_pretrain_cluster_multiseed_external_eval_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5

for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    mapfile -t MATCHES < <(ls "models/checkpoint/survival_tcga_best_hdp_pretrain_cluster_int1500_stg_r_growth8_fold${FOLD}of${N_FOLDS}_seed${SEED}_light.pt" 2>/dev/null)
    if [ "${#MATCHES[@]}" -eq 0 ]; then
      echo "[SKIP] seed=${SEED} fold=${FOLD}: checkpoint를 못 찾음 (학습이 아직 안 끝났거나 경로가 다름)"
      continue
    fi
    CKPT="${MATCHES[0]}"

    echo "=== external eval-only: seed=${SEED} fold=${FOLD} ckpt=${CKPT} Start: $(date) ==="
    python -u train_hdp_pretrain_cluster.py --dataset tcga --external --seed "${SEED}" \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" \
        --eval-external-ckpt "${CKPT}"
    echo "=== external eval-only: seed=${SEED} fold=${FOLD} Complete: $(date) ==="
  done
done
