#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-sharp-aux
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_sharpening_with_aux_pilot_seed84_fold0_%a.log

# sbatch/porpoise_attn_temperature_pilot_seed84_fold0_hpc.sh / porpoise_entropy_reg_pilot_..
# 는 둘 다 rna-aux-weight=0(no_aux 최선 레시피) 기준이었다 — 이번엔 같은 두 sharpening 방식을
# rna-aux-weight=1.0(원래 PORPOISE "full" 레시피, baseline internal C=0.7063)에도 적용해
# "aux 유무와 무관하게 sharpening이 도움이 되는가/안 되는가"까지 2x2로 채운다.
#
# array 0-1 = attn-temperature(0.2, 0.1) + aux=1.0
# array 2-3 = entropy-reg-weight(0.1, 0.3) + aux=1.0
#
# 제출: sbatch sbatch/porpoise_sharpening_with_aux_pilot_seed84_fold0_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

IDX=$SLURM_ARRAY_TASK_ID

case $IDX in
  0) EXTRA="--porpoise-attn-temperature 0.2"; TAG="temp0.2_aux1.0" ;;
  1) EXTRA="--porpoise-attn-temperature 0.1"; TAG="temp0.1_aux1.0" ;;
  2) EXTRA="--entropy-reg-weight 0.1";         TAG="entreg0.1_aux1.0" ;;
  3) EXTRA="--entropy-reg-weight 0.3";         TAG="entreg0.3_aux1.0" ;;
esac

echo "=== PORPOISE ${TAG} seed=84 fold=0/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE ${EXTRA} --rna-aux-weight 1.0 \
    --rna-genes literature_1500_intersection --dataset tcga --seed 84 \
    --backbone uni2 \
    --clinical-margin --clinical-staging \
    --patch-keep-frac 0.8 --attn-dispersion \
    --fold 0 --n-folds 5 --group-ts "0831porpoise_${TAG}"
echo "=== PORPOISE ${TAG} seed=84 fold=0/5 Complete: $(date) ==="
