#!/bin/bash
#SBATCH --job-name=PVT-fit-clusters-brca-k5
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/fit_clusters_brca_uni_k5_%j.log

# 2026-09-24: 논문 리뷰 피드백 — ClusterPool의 K=11이 임의로 고른 값인지 검증하는 ablation, BRCA
# 버전(PAAD 쪽은 fit_clusters_uni2native_k5_hpc.sh). K=11은 이미 data/cluster_centroids_brca_uni.pt
# 로 존재하므로 이 스크립트는 K=5만 새로 적합한다(K=20은 fit_clusters_brca_uni_k20_hpc.sh).
# sbatch/fit_clusters_brca_uni_hpc.sh(원래 K=11 적합, --eval-k 6 16 실루엣 탐색)와 동일 절차 —
# --k만 고정값으로 준다.
#
# k-means는 CPU 연산(sklearn MiniBatchKMeans) — --gres 지정 안 함(GPU 미할당). BRCA 실루엣
# 탐색(K=6~16)이 10~15분이었으므로 고정 K 하나는 더 빠를 것 — --time 1시간은 넉넉히.
#
# 완료 후 확인: data/cluster_centroids_brca_uni_k5.pt 생성 여부.
# 이 뒤에 sbatch/brca_m4_clusterpool_k5_multitss_bothloss_kfold_array_hpc.sh를 제출할 것.
#
# 제출: sbatch sbatch/fit_clusters_brca_uni_k5_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

echo "=== BRCA uni cluster fit (K=5) Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m data.fit_clusters_brca_uni --k 5 --max-patches-per-slide 3000 \
    --out-path data/cluster_centroids_brca_uni_k5.pt
echo "=== BRCA uni cluster fit (K=5) Complete: $(date) ==="
