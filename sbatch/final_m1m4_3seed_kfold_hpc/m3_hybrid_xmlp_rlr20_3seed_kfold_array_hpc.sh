#!/bin/bash
#SBATCH --job-name=PVT-FINAL-M3-hybrid-3seed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/final_m3_hybrid_3seed_kfold_array_%a.log

# M3 baseline(m3_3seed_kfold_array_hpc.sh, PMA WSI+RNA --no-clinical)에 이번 세션에서 검증된
# --wsi-extra-mlp + --rna-lr-mult 20.0만 추가(clinical 자체가 없어 clinical-lr-mult는 해당
# 없음) — combine_mode(concat 기본)는 baseline과 동일 유지. baseline과 나란히 돌려
# internal/external 어느 쪽이 높은지 비교 후 최종 M3로 채택할 쪽을 고른다(2026-08-16 사용자
# 결정).
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PMA_uni2native_INT1500_SS_AUX_NOCLINICAL_DISP_XMLP_RLR20 --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac --model PMA_uni2native_INT1500_SS_AUX_NOCLINICAL_DISP_XMLP_RLR20 --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/final_m1m4_3seed_kfold_hpc/m3_hybrid_xmlp_rlr20_3seed_kfold_array_hpc.sh

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

log="paper/.hpc/train_tcga_seed${SEED}_PMA_uni2native_INT1500_SS_AUX_NOCLINICAL_DISP_XMLP_RLR20_kfold5_fold${FOLD}.log"

echo "=== FINAL M3+XMLP+RLR20(PMA,WSI+RNA,no-clinical) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --no-clinical --dataset tcga --external --seed "${SEED}" \
    --rna-genes literature_1500_intersection \
    --backbone uni2native \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --wsi-extra-mlp --rna-lr-mult 20.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0816_final_m3_hybrid_3seed_kfold_hpc 2>&1 | tee "${log}"
echo "=== FINAL M3+XMLP+RLR20(PMA,WSI+RNA,no-clinical) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
