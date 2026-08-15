#!/bin/bash
#SBATCH --job-name=PVT-M2-novit-seed42-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m2_novit_seed42_kfold_array_%a.log

# 2026-08-14: M1~M4 사다리를 M4-NOVIT과 같은 WSI 아키텍처로 통일하는 작업의 M2(WSI+Clinical).
# "M1 + clinical cox_add"로 정의(사용자 지시) — self ABMIL(RNA 없음, FiLM 미적용) +
# skip-patch-vit + clinical(age/sex/margin/staging)을 cox_add로 결합. models/vit_m2.py에
# margin 지원(기존엔 없었음)과 skip_patch_vit 배선을 오늘 새로 추가했다.
#
# 완료 후(5개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 5개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M2_uni2_STG_R_DISP_COX_ADD_NOVIT --seeds 42 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m2_novit_seed42_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=42
N_FOLDS=5
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_M2_uni2_STG_R_DISP_COX_ADD_NOVIT_kfold5_fold${FOLD}.log"

echo "=== M2(uni2,cox_add,STG+R,skip-patch-vit,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M2 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --attn-dispersion \
    --skip-patch-vit \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814m2_novit_seed42_kfold5_array 2>&1 | tee "${log}"
echo "=== M2(uni2,cox_add,STG+R,skip-patch-vit,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
