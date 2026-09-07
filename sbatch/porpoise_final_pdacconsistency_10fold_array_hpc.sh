#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-final-pdaccons
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_final_pdacconsistency_10fold_array_%a.log

# 2026-09-06: sbatch/porpoise_both_loss_10fold_array_hpc.sh(최종 확정 레시피 — surv-loss both,
# nll-cox-weight 1.0. weight 스윕(0.1/0.3/3/10) 결과 1.0에서 벗어날 근거가 딱히 없었음, 사용자
# 결정)에서 RNA 유전자셋만 literature_1500_intersection(leaky, findings_backlog.md) ->
# pdac_consistency_1500(leak-free, JCI Insight 2025 5-데이터셋 교차분석 top-1500)으로 교체.
# 나머지(uni2native, CLR100, CNV, mutation, STG+R, DISP)는 완전히 동일 — RNA 유전자셋 자체의
# 효과만 분리해서 본다.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PORPOISE_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PORPOISE_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
# literature_1500_intersection 버전과 paired bootstrap 비교(--model-a/-b만 바꿔서):
#   python scripts/paired_bootstrap_delta.py --split internal --dataset tcga \
#       --model-a PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --model-b PORPOISE_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/porpoise_final_pdacconsistency_10fold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1_kfold5_fold${FOLD}.log"

echo "=== PORPOISE final recipe(pdac_consistency_1500) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906porpoise_final_pdaccons_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== PORPOISE final recipe(pdac_consistency_1500) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
