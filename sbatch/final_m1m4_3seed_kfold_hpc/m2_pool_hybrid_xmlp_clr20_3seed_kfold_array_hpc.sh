#!/bin/bash
#SBATCH --job-name=PVT-FINAL-M2POOL-hybrid-3seed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/final_m2_pool_hybrid_3seed_kfold_array_%a.log

# M2_POOL baseline(m2_pool_3seed_kfold_array_hpc.sh)에 이번 세션에서 검증된
# --wsi-extra-mlp + --clinical-lr-mult 20.0만 추가 — pooling_mode(coattn)/combine_mode(concat)는
# baseline과 동일하게 유지해 "이 두 기법의 순수 효과"만 isolate한다. baseline과 나란히 돌려
# internal/external 어느 쪽이 높은지 비교 후 최종 M2로 채택할 쪽을 고른다(2026-08-16 사용자 결정).
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M2_POOL_uni2native_SS_DISP_XMLP_CLR20 --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac --model M2_POOL_uni2native_SS_DISP_XMLP_CLR20 --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/final_m1m4_3seed_kfold_hpc/m2_pool_hybrid_xmlp_clr20_3seed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

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

log="paper/.hpc/train_tcga_seed${SEED}_M2_POOL_uni2native_SS_DISP_XMLP_CLR20_kfold5_fold${FOLD}.log"

echo "=== FINAL M2_POOL+XMLP+CLR20(coattn,concat,WSI+Clinic) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M2_POOL --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --attn-dispersion \
    --patch-keep-frac 0.8 \
    --wsi-extra-mlp --clinical-lr-mult 20.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0816_final_m2pool_hybrid_3seed_kfold_hpc 2>&1 | tee "${log}"
echo "=== FINAL M2_POOL+XMLP+CLR20(coattn,concat,WSI+Clinic) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
