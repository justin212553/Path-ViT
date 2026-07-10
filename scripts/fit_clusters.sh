#!/bin/bash
#SBATCH --job-name=PVT-KMeans
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/fit_clusters.log

# k-means 군집 중심 사전 계산 (GPU 불필요, CPU 전용)
# 출력: data/cluster_centroids.pt (K, 2048) — 기본적으로 tcga+cptac 두 코호트를 합쳐 학습
#
# 실행:
#   sbatch scripts/fit_clusters.sh                    # 기본 K=10, tcga+cptac 합산
#   sbatch scripts/fit_clusters.sh --k 16             # K 지정
#   sbatch scripts/fit_clusters.sh --dataset cptac    # cptac 코호트만
#   sbatch scripts/fit_clusters.sh --eval-k 5 20      # 실루엣 점수로 최적 K 탐색
#
# 인자는 sbatch 뒤에 붙이지 않고 아래 ARGS 변수로 지정
ARGS="--k 10"
# ARGS="--k 16"
# ARGS="--eval-k 5 20 --max-patches 3000"

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

echo "=== fit_clusters 시작: $(date) ==="
echo "인자: $ARGS"
python -u -m data.fit_clusters $ARGS
echo "=== fit_clusters 완료: $(date) ==="
