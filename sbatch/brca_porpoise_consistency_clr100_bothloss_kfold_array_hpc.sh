#!/bin/bash
#SBATCH --job-name=PVT-BRCA-porpoise-bothloss
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_porpoise_consistency_clr100_bothloss_kfold_array_%a.log

# 2026-09-07: sbatch/brca_porpoise_consistency_clr100_kfold_array_hpc.sh(BRCA_PORPOISE_CONS882_
# SS_DISP_STG_CLR100, 아직 cox 단독 loss)에서 --surv-loss both --nll-cox-weight 1.0만 추가 —
# PAAD 최종 레시피와 loss까지 맞춰서 M7/PORPOISE/ClusterPool 비교가 전부 같은 조건이 되게 한다
# (scripts/train_brca_porpoise.py에 --surv-loss 신규 이식, 사용자 지적).
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PORPOISE_CONS882_SS_DISP_STG_NLLSURV4_NLLCOX1_CLR100 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 실제 생성된 CSV로 확인 권장:
#  `ls .logs/kfold_preds/brca_BRCA_PORPOISE_CONS*NLLSURV*`)
#
# 제출: sbatch sbatch/brca_porpoise_consistency_clr100_bothloss_kfold_array_hpc.sh

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

log=".logs/train_brca_porpoise_consistency_clr100_bothloss_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA PORPOISE(consistency,STG,CLR100,both-loss) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_porpoise --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --clinical-staging --clinical-lr-mult 100 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --external-tss none --group-ts 0907_brca_porpoise_consistency_clr100_bothloss_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA PORPOISE(consistency,STG,CLR100,both-loss) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
