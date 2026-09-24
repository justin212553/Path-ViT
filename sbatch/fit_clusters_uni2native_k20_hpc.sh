#!/bin/bash
#SBATCH --job-name=PVT-fit-clusters-paad-k20
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/fit_clusters_uni2native_k20_%j.log

# 2026-09-24: fit_clusters_uni2native_k5_hpc.sh와 짝을 이루는 K=20 버전 — 헤더 설명은 그 파일
# 참조. K in {5, 11, 20} 중 K=11은 이미 data/cluster_centroids_uni2native.pt로 존재.
#
# 완료 후 확인: data/cluster_centroids_uni2native_k20.pt 생성 여부.
# 이 뒤에 sbatch/pma_clusterpool_pdaccons_k20_2seed_kfold_array_hpc.sh를 제출할 것.
#
# 제출: sbatch sbatch/fit_clusters_uni2native_k20_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

echo "=== PAAD uni2native cluster fit (K=20) Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m data.fit_clusters_uni2native --k 20 --max-patches-per-slide 3000 \
    --out-path data/cluster_centroids_uni2native_k20.pt
echo "=== PAAD uni2native cluster fit (K=20) Complete: $(date) ==="
