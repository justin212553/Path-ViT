#!/bin/bash
#SBATCH --job-name=PVT-FINAL-M4PMA-hybrid-3seed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/final_m4_pma_hybrid_3seed_kfold_array_%a.log

# M4/PMA baseline(m4_pma_3seed_kfold_array_hpc.sh, WSI+Clinic+RNA, cox_add)에 이번 세션에서
# 검증된 --wsi-extra-mlp + --clinical-lr-mult 20.0 + --rna-lr-mult 20.0을 추가 — seed42
# kfold 파일럿에서 이미 internal/external 둘 다 +0.033 개선을 확인한 조합. baseline과 나란히
# 돌려 internal/external 어느 쪽이 높은지 비교 후 최종 M4로 채택할 쪽을 고른다(2026-08-16
# 사용자 결정).
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD_XMLP_CLR20_RLR20 --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac --model PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD_XMLP_CLR20_RLR20 --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/final_m1m4_3seed_kfold_hpc/m4_pma_hybrid_xmlp_clr20_rlr20_3seed_kfold_array_hpc.sh

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

log="paper/.hpc/train_tcga_seed${SEED}_PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD_XMLP_CLR20_RLR20_kfold5_fold${FOLD}.log"

echo "=== FINAL M4/PMA+XMLP+CLR20+RLR20(WSI+Clinic+RNA,cox_add) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --dataset tcga --external --seed "${SEED}" \
    --rna-genes literature_1500_intersection \
    --backbone uni2native \
    --clinical-staging --clinical-margin \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --combine-mode cox_add \
    --wsi-extra-mlp --clinical-lr-mult 20.0 --rna-lr-mult 20.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0816_final_m4pma_hybrid_3seed_kfold_hpc 2>&1 | tee "${log}"
echo "=== FINAL M4/PMA+XMLP+CLR20+RLR20(WSI+Clinic+RNA,cox_add) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
