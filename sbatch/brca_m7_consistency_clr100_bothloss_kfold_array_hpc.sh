#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M7-bothloss
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m7_consistency_clr100_bothloss_kfold_array_%a.log

# 2026-09-07: sbatch/brca_m7_consistency_clr100_kfold_array_hpc.sh(consistency+STG+CLR100,
# 아직 cox 단독 loss)에서 --surv-loss both --nll-cox-weight 1.0만 추가 — PAAD 최종 레시피와
# loss까지 맞춰서 M7/PORPOISE/ClusterPool 세 아키텍처 비교가 전부 같은 조건이 되게 한다
# (train_light.py에 --surv-loss 신규 이식, 사용자 지적).
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_M7_CONS882_STG_NLLSURV4_NLLCOX1_CLR100 --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 model_prefix 접미사 순서를 직접 뽑은 것 — 실제 생성된 CSV로 확인 권장:
#  `ls .logs/kfold_preds/brca_BRCA_M7_CONS*NLLSURV*`)
#
# 제출: sbatch sbatch/brca_m7_consistency_clr100_bothloss_kfold_array_hpc.sh

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

log=".logs/train_brca_m7_consistency_clr100_bothloss_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M7(consistency,STG,CLR100,both-loss) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m7 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --clinical-staging --clinical-lr-mult 100 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --external-tss none --group-ts 0907_brca_m7_consistency_clr100_bothloss_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M7(consistency,STG,CLR100,both-loss) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
