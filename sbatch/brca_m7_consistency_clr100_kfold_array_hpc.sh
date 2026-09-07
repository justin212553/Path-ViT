#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M7-consistency-clr100
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m7_consistency_clr100_kfold_array_%a.log

# 2026-09-07: 기존 sbatch/brca_m7_multiseed_kfold_array_hpc.sh(BRCA_M7_TOP1500, variance
# 유전자셋, staging/CLR 없음)는 PORPOISE-BRCA 최종 레시피(BRCA_PORPOISE_CONS882_SS_DISP_STG_
# CLR100)와 유전자셋/staging/CLR이 전부 달라 공정한 대조군이 아니었다. train_brca_m7.py에
# --clinical-lr-mult 이식(scripts/train_brca_porpoise.py와 동일 관례, train.py::
# _branch_param_groups 재사용)하고, consistency RNA + staging + CLR100으로 맞춰 재실행.
#
# 배경: PAAD external에서 PORPOISE(WSI+RNA+clinical)가 M7(RNA+clinical only)을 못 이기는
# 아이러니가 나왔다(사용자 지적, 2026-09-07) — 이건 findings_backlog.md의 기존 결론(PAAD
# 규모에선 WSI 순증분이 안 잡히고, BRCA처럼 코호트가 크면 잡힌다)과 일치한다. M7의 CLR을
# 억지로 낮춰 PAAD에서 이기는 식으로 정당화하지 말고(사용자 판단: 의미 없음), 대신 WSI가
# 진짜 기여하는 걸로 이미 확인된 BRCA 쪽에서 M7 대조군을 제대로 갖추는 게 낫다는 결정.
#
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5. seed=84/126, fold=0..4.
# --external-tss none(BRCA_PORPOISE_CONS882와 동일 — 1058명 전체를 internal k-fold로).
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_M7_CONS882_STG_CLR100 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/paired_bootstrap_delta.py --split internal --dataset brca \
#       --model-a BRCA_M7_CONS882_STG_CLR100 \
#       --model-b BRCA_PORPOISE_CONS882_SS_DISP_STG_CLR100 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 CONS 유전자 수 태그는 실제 생성된 CSV로 확인 권장 —
# `ls .logs/kfold_preds/brca_BRCA_M7_CONS*` )
#
# 제출: sbatch sbatch/brca_m7_consistency_clr100_kfold_array_hpc.sh

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

log=".logs/train_brca_m7_consistency_clr100_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M7(consistency,STG,CLR100) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m7 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --clinical-staging --clinical-lr-mult 100 \
    --external-tss none --group-ts 0907_brca_m7_consistency_clr100_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M7(consistency,STG,CLR100) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
