#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M4-clusterpool-meanpool
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_clusterpool_nocoattn_multitss_kfold_array_%a.log

# 2026-09-24: PAAD 쪽 pma_clusterpool_pdaccons_nocoattn_meanpool_1seed_kfold_array_hpc.sh와
# 짝을 이루는 BRCA mean-pool ablation. scripts/train_brca_m4.py에 --no-coattn을 이번에 처음
# 이식했다(2026-09-24, train.py --no-coattn과 동일 의미 — models/vit_pma.py ViT_PMA
# use_coattn=False).
#
# sbatch/brca_m4_clusterpool_multitss_bothloss_kfold_array_hpc.sh(M4)에서 --no-coattn 하나만
# 추가 — 나머지 레시피는 완전히 동일.
#
# 2026-09-24(2차, 시간 절약 — 사용자 지시): mean-pool도 K 민감도와 같은 방향성 확인용
# ablation이라 2seed가 아니라 seed=84 1개만 돌린다.
#
# 1seed(84) x 5fold = 5 array task.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_NOCOATTN_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1 --seeds 84 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_NOCOATTN_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1 --seeds 84 --n-folds 5 --bootstrap 2000
# (정확한 태그는 실제 생성된 CSV로 확인 권장: `ls .logs/kfold_preds/brca_BRCA_PMA_CONS*NOCOATTN*`)
#
# 제출: sbatch sbatch/brca_m4_clusterpool_nocoattn_meanpool_multitss_bothloss_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_brca_m4_clusterpool_nocoattn_multitss_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M4 ClusterPool(no-coattn/mean-pool, multi-TSS external) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --clinical-staging --cluster-pool --no-coattn --clinical-lr-mult 100 \
    --external-tss multi \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --group-ts 0924_brca_m4_clusterpool_nocoattn_multitss_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M4 ClusterPool(no-coattn/mean-pool, multi-TSS external) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
