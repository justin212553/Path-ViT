#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-coxmatched-10fold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_cox_loss_matched_10fold_array_%a.log

# 2026-09-06: sbatch/porpoise_nll_surv_loss_10fold_array_hpc.sh와 **글자 하나까지 동일**하되
# --surv-loss nll_surv --nll-n-bins 4만 뺐다(기본값 cox로 돌아감) — loss 함수 하나만 딱 분리해서
# 비교하기 위한 매칭 기준선. 그 실행은 backbone(uni2native)/CLR100/CNV/mutation 전부 옛날 cox
# "no_aux" 레시피(sbatch/porpoise_no_aux_multiseed_kfold_array_hpc.sh, backbone uni2, CLR100/
# CNV/mutation 없음)와 달라서 loss만의 효과를 분리할 수 없었다 — 이 스크립트가 그 gap을 메운다.
#
# 완료 후, 이 태그와 nll_surv 태그를 paired bootstrap으로 직접 비교:
#   python scripts/paired_bootstrap_delta.py --split internal --dataset tcga \
#       --model-a PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100 \
#       --model-b PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/paired_bootstrap_delta.py --split external --dataset cptac \
#       --model-a PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100 \
#       --model-b PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
# (태그는 train.py model_prefix 체인을 직접 대조해 뽑은 것 — 혹시 안 맞으면
# `ls .logs/kfold_preds/tcga_PORPOISE*CLR100*`로 실제 파일명 확인할 것, NLLSURV4가 안 붙은
# 쪽이 이 스크립트 결과.)
#
# 제출: sbatch sbatch/porpoise_cox_loss_matched_10fold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_kfold5_fold${FOLD}.log"

echo "=== PORPOISE cox loss(uni2native,CLR100,CNV,MUT,STG+R) — nll_surv 매칭 기준선 — seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906porpoise_coxmatched_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== PORPOISE cox loss(uni2native,CLR100,CNV,MUT,STG+R) — nll_surv 매칭 기준선 — seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
