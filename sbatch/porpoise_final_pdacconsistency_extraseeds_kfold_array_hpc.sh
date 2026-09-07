#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-extraseeds-pdaccons
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_final_pdacconsistency_extraseeds_kfold_array_%a.log

# 2026-09-06: sbatch/porpoise_final_pdacconsistency_10fold_array_hpc.sh(pdac_consistency_1500,
# seed 84/126)와 완전히 동일한 레시피를 seed 168/210/252로 3개 더 돌린다. literature_1500 쪽
# 동일 목적의 추가 시드는 sbatch/porpoise_both_loss_extraseeds_int1500_kfold_array_hpc.sh —
# 그쪽 주석에 전체 배경(왜 시드를 늘리는지, seed42를 왜 뺐는지) 설명 있음.
#
# 완료 후, 기존 84/126과 합쳐 5seed 기준으로 재pooling:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PORPOISE_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,168,210,252 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PORPOISE_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,168,210,252 --n-folds 5 --bootstrap 2000
# literature_1500 쪽(동일하게 5seed로 pooling한 뒤)과 paired bootstrap 재비교:
#   python scripts/paired_bootstrap_delta.py --split internal --dataset tcga \
#       --model-a PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --model-b PORPOISE_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,168,210,252 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/porpoise_final_pdacconsistency_extraseeds_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(168 210 252)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1_kfold5_fold${FOLD}.log"

echo "=== PORPOISE final recipe(pdac_consistency_1500) extra-seed seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906porpoise_final_pdaccons_extraseeds_kfold5_array 2>&1 | tee "${log}"
echo "=== PORPOISE final recipe(pdac_consistency_1500) extra-seed seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
