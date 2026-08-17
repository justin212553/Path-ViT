#!/bin/bash
#SBATCH --job-name=PVT-FINAL-M2POOL-hybrid-retry
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/final_m2_pool_hybrid_seed126_fold3_retry.log

# m2_pool_hybrid_xmlp_clr20_3seed_kfold_array_hpc.sh의 array index 13(seed126,fold3)만 누락돼
# 단일 재실행(2026-08-16, 원인 미상 — 로그가 mkdir 버그로 사라져서 실패 사유 확인 불가).

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

log="paper/.hpc/train_tcga_seed126_M2_POOL_uni2native_SS_DISP_XMLP_CLR20_kfold5_fold3.log"

echo "=== FINAL M2_POOL+XMLP+CLR20 seed=126 fold=3/5 RETRY Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M2_POOL --dataset tcga --external --seed 126 \
    --backbone uni2native \
    --attn-dispersion \
    --patch-keep-frac 0.8 \
    --wsi-extra-mlp --clinical-lr-mult 20.0 \
    --fold 3 --n-folds 5 --group-ts 0816_final_m2pool_hybrid_3seed_kfold_hpc 2>&1 | tee "${log}"
echo "=== FINAL M2_POOL+XMLP+CLR20 seed=126 fold=3/5 RETRY Complete: $(date) ==="
