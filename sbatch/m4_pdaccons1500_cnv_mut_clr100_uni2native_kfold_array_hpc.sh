#!/bin/bash
#SBATCH --job-name=PVT-M4-pdaccons-uni2native
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m4_pdaccons1500_cnv_mut_clr100_uni2native_kfold_array_%a.log

# 2026-09-05: 오늘 로컬(단일 fold0/seed84)로 확인한 "최고사양" M4 레시피(pdac_consistency_1500
# +CNV+mutation+staging+margin+CLR100+cox_add)를 2seed x 5fold로 정식 재현 — 이번엔
# --backbone uni2native를 명시적으로 박는다. 오늘 세션 안에서 만든 다른 실험 스크립트들
# (experiment_m4_wsi_cox_add.py 등)이 --backbone을 안 넘겨 조용히 기본값(resnet50)으로
# 돌아간 걸 뒤늦게 발견한 사고(project_wsi_weak_contribution_investigation 메모 참조)가
# 있었음 — 이 sbatch는 그 실수를 안 반복하도록 처음부터 명시.
#
# 로컬 단일 fold0/seed84 참고 수치(--backbone uni2native, 오늘 확인):
#   external c_index = 0.630 (CLR100)
#
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5. seed=84/126, fold=0..4.
# --requeue: free-gpu partition preemption 대비.
#
# 완료 후(10개 fold 로그 확인, .logs/kfold_preds/에 CSV 10개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M4_uni2native_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_CLR100_LRMW10 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model M4_uni2native_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_CLR100_LRMW10 --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m4_pdaccons1500_cnv_mut_clr100_uni2native_kfold_array_hpc.sh

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

log=".logs/train_tcga_m4_pdaccons1500_cnv_mut_clr100_uni2native_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M4(pdac_consistency_1500+CNV+mutation+STG+R+CLR100, uni2native) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native --rna-genes pdac_consistency_1500 --use-cnv --clinical-mutation \
    --clinical-staging --clinical-margin --combine-mode cox_add \
    --clinical-lr-mult 100 --lr-mult-warmup-epochs 10 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --group-ts 0905_m4_pdaccons1500_cnv_mut_clr100_uni2native_kfold_array 2>&1 | tee "${log}"
echo "=== M4(pdac_consistency_1500+CNV+mutation+STG+R+CLR100, uni2native) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
