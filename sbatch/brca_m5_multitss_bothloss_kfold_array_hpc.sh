#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M5-multitss
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m5_multitss_kfold_array_%a.log

# 2026-09-10: BRCA M1~M7 multi-TSS external test(sbatch/brca_m1_..._hpc.sh 헤더 참조) — M5
# (Clinical only, WSI/RNA 없음, scripts/train_brca_m5.py 신규).
#
# 2seed(84,126) x 5fold.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_M5_STG_NLLSURV4_NLLCOX1 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset brca \
#       --model BRCA_M5_STG_NLLSURV4_NLLCOX1 --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 실제 생성된 CSV로 확인 권장: `ls .logs/kfold_preds/brca_BRCA_M5*`)
#
# 제출: sbatch sbatch/brca_m5_multitss_bothloss_kfold_array_hpc.sh

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

log=".logs/train_brca_m5_multitss_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M5(multi-TSS external) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m5 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --clinical-staging \
    --external-tss multi \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --group-ts 0910_brca_m5_multitss_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M5(multi-TSS external) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
