#!/bin/bash
#SBATCH --job-name=PVT-attn-alt-test
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/attn_alt_test_fold0seed84.log

# 2026-09-05: 나이스트롬 대신/보완할 3가지 패치 간 정보교환 방식 — 한 fold(0)/seed(84)씩만
# 순서대로(동시 아님, GPU 1장 공유) 빠르게 비교. paper/test/attn_test*_fold0seed84.log가
# GPU 미할당(로그인 노드에서 직접 실행)으로 실패한 뒤 재시도하는 버전 — sbatch로 A30 확보.
#
# 비교 기준(오늘 이미 확인된 같은 레시피의 baseline, 단일 fold0/seed84, Nystrom 기본값):
#   internal test c_index = 0.536, external c_index = 0.630 (CLR100)
#
# 제출: sbatch sbatch/attn_alternatives_test_fold0seed84_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

BASE_ARGS="--M4 --dataset tcga --external --seed 84 --fold 0 --n-folds 5 \
  --backbone uni2native --rna-genes pdac_consistency_1500 --use-cnv --clinical-mutation \
  --clinical-staging --clinical-margin --combine-mode cox_add \
  --clinical-lr-mult 100 --lr-mult-warmup-epochs 10"

echo "=== 1) Nystrom landmark 버그 수정 (--fix-nystrom-landmarks) Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py $BASE_ARGS --fix-nystrom-landmarks 2>&1 | tee .logs/attn_test1_nystromfix_fold0seed84.log
echo "=== 1) Complete: $(date) ==="

echo "=== 2) kNN 평균 집계, attention 없음 (--knn-mean-agg) Start: $(date) ==="
python -u ./train.py $BASE_ARGS --knn-mean-agg 2>&1 | tee .logs/attn_test2_knnmeanagg_fold0seed84.log
echo "=== 2) Complete: $(date) ==="

echo "=== 3) 클러스터 압축 + 슈퍼토큰 dense attention (--cluster-attn) Start: $(date) ==="
python -u ./train.py $BASE_ARGS --cluster-attn --n-clusters 16 2>&1 | tee .logs/attn_test3_clusterattn_fold0seed84.log
echo "=== 3) Complete: $(date) ==="
