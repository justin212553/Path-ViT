#!/bin/bash
#SBATCH --job-name=PVT-M4-clusterpool-k20
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_clusterpool_pdaccons_k20_kfold_array_%a.log

# 2026-09-24: pma_clusterpool_pdaccons_k5_1seed_kfold_array_hpc.sh와 짝을 이루는 K=20 버전 —
# 헤더 설명(1seed로 줄인 이유 포함)은 그 파일 참조. 먼저 sbatch/fit_clusters_uni2native_k20_hpc.sh
# 로 data/cluster_centroids_uni2native_k20.pt를 만들어야 한다.
#
# 1seed(84) x 5fold = 5 array task.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_CLUSTERPOOL_CENTROIDSCLUSTER_CENTROIDS_UNI2NATIVE_K20_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_CLUSTERPOOL_CENTROIDSCLUSTER_CENTROIDS_UNI2NATIVE_K20_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84 --n-folds 5 --bootstrap 2000
# (정확한 접미사 순서는 추정치 — 실제 생성된 CSV로 확인 권장:
#  `ls .logs/kfold_preds/tcga_PMA_uni2native_PDACCONS1500*CENTROIDS*K20*`)
#
# 제출: sbatch sbatch/pma_clusterpool_pdaccons_k20_1seed_kfold_array_hpc.sh

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

log=".logs/train_tcga_m4_clusterpool_k20_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M4(PMA ClusterPool, K=20, 1seed) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --cluster-pool --cluster-centroids-path data/cluster_centroids_uni2native_k20.pt \
    --combine-mode cox_add --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0924pma_clusterpool_pdaccons_k20_1seed_kfold5_array 2>&1 | tee "${log}"
echo "=== M4(PMA ClusterPool, K=20, 1seed) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
