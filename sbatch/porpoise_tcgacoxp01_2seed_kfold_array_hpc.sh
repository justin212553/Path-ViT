#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-tcgacoxp01
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_tcgacoxp01_2seed_kfold_array_%a.log

# 2026-09-07: TCGA-only Cox(nominal p<0.01, 719개, CPTAC 라벨 전혀 미참조)를 단독 RNA
# 세트로 학습(--rna-genes tcga_cox_p01, 신규) — 확정 최종 레시피에서 RNA 세트만 교체,
# 나머지 전부 동일.
#
# 목적: pdac_consistency_1500(generalist, TCGA/CPTAC 둘 다 라벨 미참조)과 예측을 앙상블할
# "domain-specific(TCGA에 맞춰진)" 짝 — literature_1500_intersection을 이 역할로 쓰면
# 그 자체의 CPTAC-only 순위 참조(leakage)가 앙상블에도 그대로 섞여 방어가 안 되는 문제를
# 피하기 위함(사용자 지적, 2026-09-06/07). 이 유전자셋은 TCGA 라벨만 보므로 CPTAC 쪽은
# 여전히 완전 미노출 — external은 진짜 held-out.
#
# 2시드(84,126)로 시작 — 새 RNA 세트라 방향성부터 확인.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PORPOISE_uni2native_TCGACOXP01_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PORPOISE_uni2native_TCGACOXP01_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
# pdac_consistency_1500과 앙상블(2seed로 맞춰서, scripts/pool_cross_genes_ensemble.py):
#   python scripts/pool_cross_genes_ensemble.py --dataset cptac --split external \
#       --models PORPOISE_uni2native_TCGACOXP01_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1,PORPOISE_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/porpoise_tcgacoxp01_2seed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2native_TCGACOXP01_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1_kfold5_fold${FOLD}.log"

echo "=== PORPOISE final recipe(tcga_cox_p01) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes tcga_cox_p01 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0907porpoise_tcgacoxp01_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== PORPOISE final recipe(tcga_cox_p01) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
