#!/bin/bash
#SBATCH --job-name=PVT-M1POOL-uni2-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logks/m1_pool_uni2_multiseed_kfold_array_%a.log

# M1_POOL(다성분 pooling+self-attention, WSI 단독, clinical/RNA 없음)을 UNI2-h로 3seed(42/84/126)
# x 5-fold = 15개 array job으로 학습. sbatch/pma_uni2_coxadd_stg_multiseed_kfold_array_hpc.sh와
# 동일한 seed/fold 매핑 관례(IDX -> seed_idx=IDX/5, fold=IDX%5). precomputed features_uni2.pt
# 사용(--image 불필요, 빠름). M1(ABMIL)/M2(ABMIL)가 UNI로 external 0.5를 못 넘겨서
# self-attention pooling으로 바꾼 버전 — M1~M7+PMA 전체를 UNI2 기준으로 다시 맞추는 세션의 일부.
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M1_POOL_uni2_SS_DISP --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m1_pool_uni2_multiseed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(42 84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_M1_POOL_uni2_SS_DISP_kfold5_fold${FOLD}.log"

echo "=== M1_POOL(uni2) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M1_POOL --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --patch-keep-frac 0.8 --attn-dispersion \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0808m1pool_uni2_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== M1_POOL(uni2) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
