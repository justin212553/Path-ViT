#!/bin/bash
#SBATCH --job-name=PVT-BRCA-cp-clr100-bothloss
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_clusterpool_clr100_bothloss_kfold_array_%a.log

# 2026-09-07: sbatch/brca_m4_clusterpool_clr100_kfold_array_hpc.sh(BRCA_PMA_CONS882_STG_SS_AUX_
# CLUSTERPOOL_CLR100 — M7을 처음 이긴 조합, 아직 cox 단독 loss)에서 --surv-loss both
# --nll-cox-weight 1.0만 추가 — PAAD 최종 레시피와 loss까지 맞춰서 재검증한다. M7/PORPOISE는
# 매칭 loss로 각각 BRCA에서 유의한 WSI 기여를 못 보였으니(2026-09-07), ClusterPool이 이
# 매칭 조건에서도 여전히 유의하게 이기는지가 핵심 질문.
#
# --rna-aux-weight는 train_brca_m4.py 기본값(1.0, PAAD와 다른 BRCA 자체 관례) 그대로 유지 —
# 기존 CLUSTERPOOL 비교와 AUX 유무까지 바뀌면 안 되므로 명시적으로 안 건드림.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_SS_AUX_CLUSTERPOOL_NLLSURV4_NLLCOX1_CLR100 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/paired_bootstrap_delta.py --split internal --dataset brca \
#       --model-a BRCA_M7_CONS882_STG_NLLSURV4_NLLCOX1_CLR100 \
#       --model-b BRCA_PMA_CONS882_STG_SS_AUX_CLUSTERPOOL_NLLSURV4_NLLCOX1_CLR100 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 실제 생성된 CSV로 확인 권장:
#  `ls .logs/kfold_preds/brca_BRCA_PMA_CONS*NLLSURV*`)
#
# 제출: sbatch sbatch/brca_m4_clusterpool_clr100_bothloss_kfold_array_hpc.sh

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

log=".logs/train_brca_m4_clusterpool_clr100_bothloss_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA cluster_pool+CLR100+both-loss seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --clinical-staging --cluster-pool --clinical-lr-mult 100 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --external-tss none --group-ts 0907_brca_m4_clusterpool_clr100_bothloss_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA cluster_pool+CLR100+both-loss seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
