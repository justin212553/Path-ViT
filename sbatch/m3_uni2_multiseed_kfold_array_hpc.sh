#!/bin/bash
#SBATCH --job-name=PVT-M3-uni2-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m3_uni2_multiseed_kfold_array_%a.log

# M3(다성분 pooling + RNA co-attention, WSI+RNA만 — clinical 없음, --PMA --no-clinical)을
# UNI2-h로 3seed(42/84/126) x 5-fold = 15개 array job으로 학습. m1/m2_pool과 동일한 seed/fold
# 매핑 관례. precomputed features_uni2.pt 사용(--image 불필요).
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2_INT1500_SS_AUX_NOCLINICAL_DISP --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m3_uni2_multiseed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PMA_uni2_INT1500_SS_AUX_NOCLINICAL_DISP_kfold5_fold${FOLD}.log"

echo "=== M3(uni2) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --no-clinical --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0808m3_uni2_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== M3(uni2) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
