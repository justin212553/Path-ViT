#!/bin/bash
#SBATCH --job-name=PVT-M1M2POOL-xmlp-seed42-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m1pool_m2pool_xmlp_seed42_kfold_array_%a.log

# M1_POOL(다성분 pooling+self-attention, WSI 단독)/M2_POOL(M1_POOL+clinical cox_add,
# selfattn 유지)에 --wsi-extra-mlp(M2_POOL은 --clinical-lr-mult 20.0도)를 이식한 뒤 seed42로
# 확정 짓는 마지막 두 모델. 로컬 fold0 스모크테스트(2026-08-16)는 크래시 없이 정상 동작 확인
# 완료 — external은 M1_POOL 0.4891/M2_POOL 0.4743로 XMLP 없이 돌린 것과 마찬가지로 여전히
# chance 근처(예상된 결과, WSI self-attention pooling 단독으로는 신호가 없다는 이전 결론과 일치).
# array index 0-4 = M1_POOL fold0-4, 5-9 = M2_POOL fold0-4.
#
# 완료 후:
#   python scripts/pool_kfold_preds.py --dataset tcga --model M1_POOL_uni2_SS_DISP_XMLP --seed 42 --n-folds 5
#   python scripts/pool_kfold_preds.py --dataset tcga --model M2_POOL_uni2_SS_R_DISP_COX_ADD_SELFATTN_XMLP_CLR20 --seed 42 --n-folds 5
#
# 제출: sbatch sbatch/m1pool_m2pool_xmlp_seed42_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=42
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID

COMMON="--backbone uni2 --attn-dispersion --patch-keep-frac 0.8 --wsi-extra-mlp --dataset tcga --external --seed ${SEED} --n-folds ${N_FOLDS} --group-ts 0816_m1m2pool_xmlp_seed42_kfold_array"

if [ "$IDX" -lt 5 ]; then
    FOLD=$IDX
    log=".logs/train_tcga_seed${SEED}_M1_POOL_uni2_SS_DISP_XMLP_kfold5_fold${FOLD}.log"
    echo "=== M1_POOL(selfattn,XMLP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
    python -u ./train.py --M1_POOL ${COMMON} --fold "${FOLD}" 2>&1 | tee "${log}"
    echo "=== M1_POOL(selfattn,XMLP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
else
    FOLD=$((IDX - 5))
    log=".logs/train_tcga_seed${SEED}_M2_POOL_uni2_SS_R_DISP_COX_ADD_SELFATTN_XMLP_CLR20_kfold5_fold${FOLD}.log"
    echo "=== M2_POOL(selfattn,cox_add,XMLP,CLR20) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
    python -u ./train.py --M2_POOL --pooling-mode selfattn --combine-mode cox_add --clinical-margin \
        --clinical-lr-mult 20.0 ${COMMON} --fold "${FOLD}" 2>&1 | tee "${log}"
    echo "=== M2_POOL(selfattn,cox_add,XMLP,CLR20) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
fi
