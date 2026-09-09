#!/bin/bash
#SBATCH --job-name=PVT-M1-clusterpool
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m1_clusterpool_final_recipe_kfold_array_%a.log

# 2026-09-09: 논문 M1~M7 전체 ablation 테이블을 최종 레시피로 재작성하기 위한 재실행 중 하나.
# M1(WSI only, clinical/RNA 없음)에 ClusterPool을 새로 이식(models/vit_m1.py, 2026-09-09 —
# RNA가 없어 --PMA의 RNA-guided co-attention을 못 쓰므로 query 없는 self-attention pooling
# models/self_attention_pooling.py::SelfAttentionPooling을 대신 쓴다)해서 M4(=--PMA
# --cluster-pool)와 pooling 메커니즘을 통일한다. RNA/Clinical이 아예 없는 모델이라 gene-set/
# staging/margin/mutation/CLR 전부 해당 없음 — cluster-pool + backbone + patch-keep-frac +
# loss만 M4와 맞춘다.
#
# 2seed(84,126) x 5fold — 이 프로젝트 표준.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M1_uni2native_CLUSTERPOOL_NLLSURV4_NLLCOX1 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model M1_uni2native_CLUSTERPOOL_NLLSURV4_NLLCOX1 --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 실제 생성된 CSV로 확인 권장: `ls .logs/kfold_preds/tcga_M1_uni2native*CLUSTERPOOL*`)
#
# 제출: sbatch sbatch/m1_clusterpool_final_recipe_2seed_kfold_array_hpc.sh

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

log=".logs/train_tcga_m1_clusterpool_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M1 ClusterPool(확정 레시피) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M1 --cluster-pool --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --patch-keep-frac 0.8 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0909m1_clusterpool_final_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== M1 ClusterPool(확정 레시피) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
