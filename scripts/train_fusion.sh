#!/bin/bash
#SBATCH --job-name=PVT-Fusion
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_fusion.log

# LateFusionViT 학습 (ViT+ABMIL + Cluster Histogram Branch)
# 선행 조건: cluster_centroids.pt 생성 완료 (scripts/fit_clusters.sh)
#
# ablation 비교:
#   baseline : bash scripts/train.sh            → survival_{dataset}_best.pt (tcga/cptac 둘 다 제출)
#   fusion   : sbatch scripts/train_fusion.sh   → survival_{dataset}_best_fusion.pt

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

# cluster_centroids.pt 존재 확인
if [ ! -f cluster_centroids.pt ]; then
    echo "오류: cluster_centroids.pt 없음 — fit_clusters.sh를 먼저 실행하세요."
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== LateFusionViT 학습 시작: $(date) ==="
python -u ./train.py --fusion
echo "=== LateFusionViT 학습 완료: $(date) ==="
