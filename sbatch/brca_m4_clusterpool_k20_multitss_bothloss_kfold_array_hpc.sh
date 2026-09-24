#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M4-clusterpool-k20
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_clusterpool_k20_multitss_kfold_array_%a.log

# 2026-09-24: brca_m4_clusterpool_k5_multitss_bothloss_kfold_array_hpc.sh와 짝을 이루는 K=20
# 버전 — 헤더 설명(1seed로 줄인 이유 포함)은 그 파일 참조. 먼저
# sbatch/fit_clusters_brca_uni_k20_hpc.sh로 data/cluster_centroids_brca_uni_k20.pt를 만들어야
# 한다.
#
# 1seed(84) x 5fold = 5 array task.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_CLUSTERPOOL_CENTROIDSCLUSTER_CENTROIDS_BRCA_UNI_K20_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_CLUSTERPOOL_CENTROIDSCLUSTER_CENTROIDS_BRCA_UNI_K20_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84 --n-folds 5 --bootstrap 2000
# (정확한 접미사 순서는 추정치 — 실제 생성된 CSV로 확인 권장:
#  `ls .logs/kfold_preds/brca_BRCA_PMA_CONS*CENTROIDS*K20*`)
#
# 제출: sbatch sbatch/brca_m4_clusterpool_k20_multitss_bothloss_kfold_array_hpc.sh

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

log=".logs/train_brca_m4_clusterpool_k20_multitss_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M4 ClusterPool(K=20, multi-TSS external) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --clinical-staging --cluster-pool \
    --cluster-centroids-path data/cluster_centroids_brca_uni_k20.pt --clinical-lr-mult 100 \
    --external-tss multi \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --group-ts 0924_brca_m4_clusterpool_k20_multitss_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M4 ClusterPool(K=20, multi-TSS external) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
