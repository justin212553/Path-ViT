#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M3-multitss-noaux
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m3_clusterpool_multitss_noaux_kfold_array_%a.log

# 2026-09-12: sbatch/brca_m3_clusterpool_multitss_bothloss_kfold_array_hpc.sh 재실행 —
# train_brca_m4.py의 --rna-aux-weight 기본값이 1.0(RNA-prediction auxiliary task 켜짐)인데,
# PAAD 최종 레시피(train.py --rna-aux-weight 기본값 0.0)는 이 항을 아예 안 쓴다. 논문을
# "파이널 레시피 기준으로만 서술"하기로 하면서 BRCA만 이 항을 몰래 더 쓰고 있는 게 드러나
# PAAD와 동일하게 --rna-aux-weight 0.0으로 꺼서 재실행한다(사용자 지시).
#
# 2seed(84,126) x 5fold. 나머지 레시피(ClusterPool, no-clinical, gene-selection consistency,
# multi-TSS external, both-loss)는 원본과 동일 유지.
#
# 완료 후 모델 태그는 기존 _AUX 접미사가 빠진 형태일 것 — 실제 생성된 CSV로 확인:
#   ls .logs/kfold_preds/brca_BRCA_PMA_CONS882_NOCLINICAL*CLUSTERPOOL*
#
# 제출: sbatch sbatch/brca_m3_clusterpool_multitss_bothloss_noaux_kfold_array_hpc.sh

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

log=".logs/train_brca_m3_clusterpool_multitss_noaux_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M3 ClusterPool(multi-TSS external, no aux) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --no-clinical --cluster-pool --rna-aux-weight 0.0 \
    --external-tss multi \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --group-ts 0912_brca_m3_clusterpool_multitss_noaux_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M3 ClusterPool(multi-TSS external, no aux) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
