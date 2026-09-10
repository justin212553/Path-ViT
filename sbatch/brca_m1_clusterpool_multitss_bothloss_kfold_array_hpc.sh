#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M1-multitss
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m1_clusterpool_multitss_kfold_array_%a.log

# 2026-09-10: BRCA M1~M7 external test 재개 — 기존 BH 단독 holdout(event rate 31.7%, 전체
# 13.8%의 ~2.3배 스큐)을 8/31에 포기하고 --external-tss none으로 후퇴했던 것을, "여러 기관을
# 합쳐서 event rate를 맞추자"는 사용자 결정으로 재시도(scripts/brca_common.py::
# EXTERNAL_TSS_MULTI = A2+AR+E9, N=230, event rate 13.9% — 전체 코호트와 거의 일치, 실측 확인).
# M1(WSI only, ClusterPool)부터 시작 — PAAD M1과 동일하게 RNA/Clinical 없는 순수 baseline.
#
# 2seed(84,126) x 5fold.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_M1_CLUSTERPOOL_NLLSURV4_NLLCOX1 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset brca \
#       --model BRCA_M1_CLUSTERPOOL_NLLSURV4_NLLCOX1 --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 실제 생성된 CSV로 확인 권장: `ls .logs/kfold_preds/brca_BRCA_M1*`)
#
# 제출: sbatch sbatch/brca_m1_clusterpool_multitss_bothloss_kfold_array_hpc.sh

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

log=".logs/train_brca_m1_clusterpool_multitss_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M1 ClusterPool(multi-TSS external) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m1_m2 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --external-tss multi \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --group-ts 0910_brca_m1_clusterpool_multitss_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M1 ClusterPool(multi-TSS external) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
