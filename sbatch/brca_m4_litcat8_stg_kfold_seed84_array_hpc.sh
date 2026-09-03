#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M4-litcat8-s84
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_litcat8_stg_seed84_array_%a.log

# 2026-09-03: brca_m4_litcat8_stg_kfold_array_hpc.sh(2seed x 5fold, array 0-9)의 seed84 전용
# 버전 — seed126은 이미 5/5 폴드 전부 완료(c-index=0.6351, M7의 0.719보다 낮게 나옴 — RNA 차원이
# 8차원으로 줄면서 WSI branch에 밀려 branch competition이 심해졌을 가능성). seed84는 free-gpu
# partition preemption으로 두 번이나 중간에 끊겨서 seed만 따로 분리해 다시 제출한다.
#
# --requeue 추가: preemption으로 죽으면 SLURM이 자동으로 큐에 다시 넣는다(수동 재제출 불필요) —
# 다만 train_brca_m4.py 자체는 fold 단위로 처음부터 다시 학습하므로(PORPOISE 공식 코드처럼
# 폴드별 완료 여부를 보고 스킵하는 로직 없음) 재시작해도 그 fold는 처음부터 다시 돈다.
#
# SLURM_ARRAY_TASK_ID(0~4) -> fold 그대로.
#
# 완료 후(.logs/kfold_preds/에 seed84 CSV 5개 다 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PMA_LITCAT8_STG_SS_AUX --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/paired_bootstrap_delta.py --split internal --dataset brca \
#       --model-a BRCA_M7_LITCAT8_STG --model-b BRCA_PMA_LITCAT8_STG_SS_AUX --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/brca_m4_litcat8_stg_kfold_seed84_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=84
N_FOLDS=5
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_brca_m4_litcat8_stg_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M4 litcat8+stg seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection literature_categorized --clinical-staging \
    --external-tss none --group-ts 0903_brca_m4_litcat8_stg_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M4 litcat8+stg seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
