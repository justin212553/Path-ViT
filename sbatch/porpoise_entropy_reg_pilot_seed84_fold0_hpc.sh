#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-entreg-sweep
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_entropy_reg_pilot_seed84_fold0_%a.log

# 2026-08-31: attn_pool의 patch attention entropy(0~1, 1=완전균등)를 Cox loss에 직접 벌점으로
# 더해 학습 중에 균등분포 붕괴를 억제하는 ablation(train.py --entropy-reg-weight, 범용 —
# attn_weights를 반환하는 모든 WSI 모델에서 동작하지만 여기선 PORPOISE로 검증).
# --porpoise-attn-temperature 스윕(sbatch/porpoise_attn_temperature_pilot_seed84_fold0_hpc.sh,
# 같이 병행)과 달리 이쪽은 "낮은 T를 알고 학습" 대신 "균등하면 손해"라는 명시적 신호를 준다는
# 점이 다르다 — 같은 문제(entropy 붕괴)에 대한 두 가지 다른 접근을 나란히 검증.
#
# no_aux 최종 레시피(dispersion 유지, aux 제거, seed84/fold0, baseline entropy_reg=0 internal
# C=0.7119)에 entropy_reg_weight만 4단계로 스윕 — array index 0/1/2/3 = 0.0(baseline 재확인
# 용)/0.05/0.1/0.3.
#
# 제출: sbatch sbatch/porpoise_entropy_reg_pilot_seed84_fold0_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

WEIGHTS=(0.0 0.05 0.1 0.3)
IDX=$SLURM_ARRAY_TASK_ID
W=${WEIGHTS[$IDX]}

echo "=== PORPOISE entropy-reg-weight=${W} seed=84 fold=0/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --entropy-reg-weight "${W}" \
    --rna-genes literature_1500_intersection --dataset tcga --seed 84 \
    --backbone uni2 \
    --clinical-margin --clinical-staging \
    --patch-keep-frac 0.8 --attn-dispersion \
    --fold 0 --n-folds 5 --group-ts "0831porpoise_entreg_${W}"
echo "=== PORPOISE entropy-reg-weight=${W} seed=84 fold=0/5 Complete: $(date) ==="
