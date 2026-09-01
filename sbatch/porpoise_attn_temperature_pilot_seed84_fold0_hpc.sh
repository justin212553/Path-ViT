#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-temp-sweep
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_attn_temperature_pilot_seed84_fold0_%a.log

# 2026-08-31: attn_pool의 softmax 이전 score를 처음부터 낮은 temperature로 학습시키는 ablation
# (models/vit_m1.py::AttentionPooling, train.py --porpoise-attn-temperature). 이미 학습된(T=1)
# 체크포인트에 재학습 없이 낮은 T를 후처리로 씌우면(--eval 전용, diagnose 스크립트) entropy는
# 낮아지지만 C-index가 계속 떨어졌다(0.699->0.673, T=1->0.02) — raw score가 T=1을 전제로
# 학습돼서, 다른 T로 재해석하면 신호뿐 아니라 노이즈까지 같이 증폭되는 것으로 추정된다. 그래서
# 이번엔 학습 자체를 그 T를 알고 하게 만들어, "raw score가 처음부터 낮은 T에서도 좋게 나오도록
# 최적화"되는지 확인한다.
#
# no_aux 최종 레시피(dispersion 유지, aux 제거, seed84/fold0, baseline T=1.0 internal C=0.7119)에
# temperature만 4단계로 스윕 — array index 0/1/2/3 = T 1.0(baseline 재확인용)/0.5/0.2/0.1.
#
# 제출: sbatch sbatch/porpoise_attn_temperature_pilot_seed84_fold0_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

TEMPS=(1.0 0.5 0.2 0.1)
IDX=$SLURM_ARRAY_TASK_ID
T=${TEMPS[$IDX]}

echo "=== PORPOISE attn-temperature=${T} seed=84 fold=0/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --porpoise-attn-temperature "${T}" \
    --rna-genes literature_1500_intersection --dataset tcga --seed 84 \
    --backbone uni2 \
    --clinical-margin --clinical-staging \
    --patch-keep-frac 0.8 --attn-dispersion \
    --fold 0 --n-folds 5 --group-ts "0831porpoise_temp_${T}"
echo "=== PORPOISE attn-temperature=${T} seed=84 fold=0/5 Complete: $(date) ==="
