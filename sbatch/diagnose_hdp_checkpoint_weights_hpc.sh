#!/bin/bash
#SBATCH --job-name=PVT-HDP-diag
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/diagnose_hdp_checkpoint_weights_%j.log

# scripts/diagnose_hdp_checkpoint_weights.py — GPU 필수 아님(frozen uni2native feature 위에
# 작은 Linear/Conv만 통과시키는 가벼운 forward-only 진단), 다만 srun 인터랙티브 세션이
# free-gpu 파티션 preemption/OOM/시간제한으로 죽은 문제(2026-09-01, 사용자 보고)를 피하려고
# 다른 job들과 동일하게 sbatch로 제출한다. 자원은 최소한만 잡음(30분/32G면 충분할 것으로 예상 —
# 코호트 전체(~150명)를 forward만 한 번씩 통과).
#
# seed/fold를 바꾸려면 아래 SEED/FOLD 두 줄만 수정(방금 완료한 2seed x 5fold 중 아무 조합이나).
#
# 제출: sbatch sbatch/diagnose_hdp_checkpoint_weights_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=84
FOLD=0
N_FOLDS=5

echo "=== HDP_Pretrain_Cluster checkpoint 진단 seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.diagnose_hdp_checkpoint_weights --dataset tcga --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}"
echo "=== Complete: $(date) ==="
