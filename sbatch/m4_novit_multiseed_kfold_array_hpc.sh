#!/bin/bash
#SBATCH --job-name=PVT-M4-novit-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m4_novit_multiseed_kfold_array_%a.log

# M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT — "WSI pooling을 단순화해도 안 바뀌는데,
# 그럼 patch-mixing transformer(self.vit, ViTEncoder 1-layer NystromAttention) 자체가
# 문제냐"를 검증하는 다음 단계 구조적 ablation(2026-08-14 논의).
#
# 지금까지 순서:
#  1) baseline PMA(4-component pooling + co-attention): internal 0.6359, external 0.6337
#  2) M4(단일 gated-ABMIL, RNA는 FiLM 게이트로만 개입, self.vit는 그대로 유지):
#     internal 0.6348, external 0.6325 — baseline과 사실상 동일(pooling 복잡도는 원인이 아님)
#  3) 이번: M4 + --skip-patch-vit로 self.vit(NystromAttention 1-layer)까지 아예 제거하고
#     UNI2 피처(forward_pooled 투영만 거친 것)를 바로 attn_pool(ABMIL)에 넣는다
#     (models/vit_m1.py::ViT_M1 skip_patch_vit 파라미터, 2026-08-11 도입 — "적은 표본으로
#     처음부터 학습하는 이 작은 patch-mixing transformer가 오히려 좋은 patch token을 흐릴 수
#     있다"는 가설의 구조적 ablation). 이미 강한 사전학습 UNI2-h 표현 위에 우리가 새로 얹는
#     학습 파라미터를 최소화하는 방향.
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m4_novit_multiseed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(42 84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT_kfold5_fold${FOLD}.log"

echo "=== M4(uni2,cox_add,STG+R,DISP,skip-patch-vit) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4 --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --skip-patch-vit \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814m4_novit_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== M4(uni2,cox_add,STG+R,DISP,skip-patch-vit) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
