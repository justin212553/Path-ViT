#!/bin/bash
#SBATCH --job-name=PVT-M4-clusterpool-k5
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_clusterpool_pdaccons_k5_kfold_array_%a.log

# 2026-09-24: 논문 리뷰 피드백 — ClusterPool의 K=11이 임의로 고른 값인지, M4 결과가 K에 얼마나
# 민감한지 검증하는 ablation. 먼저 sbatch/fit_clusters_uni2native_k5_hpc.sh로
# data/cluster_centroids_uni2native_k5.pt를 만들어야 한다(없으면 ViT_PMA가 FileNotFoundError로
# 즉시 실패).
#
# 2026-09-24(2차, 시간 절약 — 사용자 지시): K 민감도는 A(M4-SA/M3-SA)·C(시드 확장)와 달리 M4와의
# paired significance test가 목적이 아니라 "K를 바꿔도 대략 같은 자리인가"를 보는 방향성
# 확인이므로, 2seed가 아니라 seed=84 1개만 돌린다. K=20(pma_clusterpool_pdaccons_k20_1seed_
# kfold_array_hpc.sh)도 동일하게 1seed.
#
# sbatch/pma_clusterpool_pdaccons_final_recipe_2seed_kfold_array_hpc.sh(M4, K=11 기본값)에서
# --cluster-centroids-path만 K=5 centroids로 바꾸고 seed를 84 하나로 줄인 것 — 나머지 레시피는
# 완전히 동일.
#
# 2026-09-24(버그 예방, train.py 수정): --cluster-centroids-path는 원래 model_prefix/tag에
# 전혀 반영되지 않아, 이 스크립트를 K=11 기본 실행과 같은 이름으로 덮어쓸 뻔했다 — train.py에
# _CENTROIDS{경로 stem 대문자} 접미사를 추가해 원천 차단했다(2026-09-24). 이 경로
# (data/cluster_centroids_uni2native_k5.pt)의 stem은 "cluster_centroids_uni2native_k5"라
# 태그에 _CENTROIDSCLUSTER_CENTROIDS_UNI2NATIVE_K5가 자동으로 붙는다.
#
# 1seed(84) x 5fold = 5 array task.
#
# 완료 후(단일 시드라 pool_multiseed_kfold_preds.py --seeds 84 하나만 줘도 동작함 — bootstrap
# CI는 fold 간 변이만 반영, seed 변이는 반영 안 됨에 유의):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_CLUSTERPOOL_CENTROIDSCLUSTER_CENTROIDS_UNI2NATIVE_K5_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_CLUSTERPOOL_CENTROIDSCLUSTER_CENTROIDS_UNI2NATIVE_K5_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84 --n-folds 5 --bootstrap 2000
# (정확한 접미사 순서는 추정치 — 실제 생성된 CSV로 확인 권장:
#  `ls .logs/kfold_preds/tcga_PMA_uni2native_PDACCONS1500*CENTROIDS*K5*`)
#
# 제출: sbatch sbatch/pma_clusterpool_pdaccons_k5_1seed_kfold_array_hpc.sh

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

log=".logs/train_tcga_m4_clusterpool_k5_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M4(PMA ClusterPool, K=5, 1seed) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --cluster-pool --cluster-centroids-path data/cluster_centroids_uni2native_k5.pt \
    --combine-mode cox_add --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0924pma_clusterpool_pdaccons_k5_1seed_kfold5_array 2>&1 | tee "${log}"
echo "=== M4(PMA ClusterPool, K=5, 1seed) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
