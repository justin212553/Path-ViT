#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M4-clusterpool-k5
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_clusterpool_k5_multitss_kfold_array_%a.log

# 2026-09-24: PAAD 쪽 pma_clusterpool_pdaccons_k5_1seed_kfold_array_hpc.sh와 짝을 이루는 BRCA
# K=5 버전 — ClusterPool의 K=11이 임의로 고른 값인지 검증. 먼저
# sbatch/fit_clusters_brca_uni_k5_hpc.sh로 data/cluster_centroids_brca_uni_k5.pt를 만들어야
# 한다.
#
# sbatch/brca_m4_clusterpool_multitss_bothloss_kfold_array_hpc.sh(M4, K=11 기본값)에서
# --cluster-centroids-path만 K=5로 바꾼 것 — 나머지 레시피는 완전히 동일.
#
# 2026-09-24(버그 예방, scripts/train_brca_m4.py 수정): 기본 경로(data/cluster_centroids_
# brca_uni.pt)가 아닌 --cluster-centroids-path를 쓰면 태그에 _CENTROIDS{stem 대문자}가
# 자동으로 붙는다(train.py와 동일 수정) — 이 경로의 stem은 "cluster_centroids_brca_uni_k5".
#
# 2026-09-24(2차, 시간 절약 — 사용자 지시): K 민감도는 paired significance가 목적이 아니라
# 방향성 확인이므로 2seed가 아니라 seed=84 1개만 돌린다(PAAD와 동일 판단).
#
# 1seed(84) x 5fold = 5 array task.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_CLUSTERPOOL_CENTROIDSCLUSTER_CENTROIDS_BRCA_UNI_K5_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_CLUSTERPOOL_CENTROIDSCLUSTER_CENTROIDS_BRCA_UNI_K5_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84 --n-folds 5 --bootstrap 2000
# (정확한 접미사 순서는 추정치 — 실제 생성된 CSV로 확인 권장:
#  `ls .logs/kfold_preds/brca_BRCA_PMA_CONS*CENTROIDS*K5*`)
#
# 제출: sbatch sbatch/brca_m4_clusterpool_k5_multitss_bothloss_kfold_array_hpc.sh

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

log=".logs/train_brca_m4_clusterpool_k5_multitss_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M4 ClusterPool(K=5, multi-TSS external) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --clinical-staging --cluster-pool \
    --cluster-centroids-path data/cluster_centroids_brca_uni_k5.pt --clinical-lr-mult 100 \
    --external-tss multi \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --group-ts 0924_brca_m4_clusterpool_k5_multitss_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M4 ClusterPool(K=5, multi-TSS external) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
