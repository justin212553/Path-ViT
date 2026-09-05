#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M4-vit2es10
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_consistency_stg_vitlayers2_es10_kfold_array_%a.log

# 2026-09-05: brca_m4_consistency_stg_kfold_array_hpc.sh(1-layer 기본, CONS882+STG+SS+AUX, 이미
# 10-fold 전부 완료됨 - .logs/kfold_preds/brca_BRCA_PMA_CONS882_STG_SS_AUX_seed{84,126}_fold{0-4}of5.csv)의
# 2-layer + early-stop 변형. 로컬 fold0/seed84 단일 실행에서 val_c_index는 1-layer 대비 뚜렷이
# 올랐지만(0.72), test_c_index=0.6691은 오히려 같은 fold/seed의 M7(RNA+clinical only, WSI
# 없음, test_c_index=0.7367)보다 낮았다 — WSI가 여전히 기여하지 못하는(PAAD와 동일한) 패턴이
# BRCA(N=1058)에서도 재현되는지, 아니면 fold0/seed84 하나만의 우연인지 10-fold 전체로 확인 필요.
#
# 로컬에서 이미 확인된 정확한 레시피(scripts/train_brca_m4.py, wandb run명
# BRCA_BRCA_PMA_CONS882_STG_SS_AUX_VITLAYERS2_ES10_seed84_fold0of5로 확인):
#   --gene-selection consistency --clinical-staging --num-transformer-layers 2
#   --early-stop-patience 10 --epochs 200 --external-tss none
# (patch-keep-frac 0.8, rna-aux-weight 1.0은 기본값 그대로 — 이미 SS/AUX 태그로 반영됨)
#
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5. seed=84/126, fold=0..4.
# --requeue: free-gpu partition preemption 대비. --time=24:00:00: 로컬 fold0/seed84가
# early-stop(epoch 59)까지 1h31m 걸렸으니 넉넉히 잡음.
#
# 완료 후(10개 fold 로그 확인, .logs/kfold_preds/에 CSV 10개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_SS_AUX_VITLAYERS2_ES10 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   M7(1-layer 스크립트와 동일 M7, .logs/kfold_preds/brca_BRCA_M7_CONS882_STG_seed{84,126}_fold{0-4}of5.csv,
#   이미 완료됨)과의 paired 비교:
#   python scripts/paired_bootstrap_delta.py --split internal --dataset brca \
#       --model-a BRCA_M7_CONS882_STG \
#       --model-b BRCA_PMA_CONS882_STG_SS_AUX_VITLAYERS2_ES10 --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/brca_m4_consistency_stg_vitlayers2_es10_kfold_array_hpc.sh

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

log=".logs/train_brca_m4_consistency_stg_vitlayers2_es10_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M4 consistency+stg+VITLAYERS2+ES10 seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --clinical-staging \
    --num-transformer-layers 2 --early-stop-patience 10 --epochs 200 \
    --external-tss none --group-ts 0905_brca_m4_consistency_stg_vitlayers2_es10_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M4 consistency+stg+VITLAYERS2+ES10 seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
