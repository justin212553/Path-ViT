#!/bin/bash
#SBATCH --job-name=PVT-fit-clusters-paad-k5
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/fit_clusters_uni2native_k5_%j.log

# 2026-09-24: 논문 리뷰 피드백 — ClusterPool의 K=11이 임의로 고른 값인지, 결과가 K에 민감한지
# 검증하는 ablation 1단계. K in {5, 11, 20} 중 K=11은 이미 data/cluster_centroids_uni2native.pt로
# 존재하므로, 이 스크립트는 K=5만 별도 파일로 새로 적합한다(K=20은
# fit_clusters_uni2native_k20_hpc.sh). data/fit_clusters_uni2native.py와 완전히 동일한
# 절차(TCGA-PAAD 학습 코호트 패치에서만 MiniBatchKMeans, CPTAC 미참조)이며 --k만 고정값으로 준다
# (--eval-k 실루엣 탐색은 안 씀 — K 자체를 실험 변수로 고정하고 싶은 것이므로).
#
# k-means 자체는 GPU가 필요 없는 CPU 연산(sklearn MiniBatchKMeans) — free-gpu 파티션은 대기시간이
# 짧아서 그대로 쓰되 --gres는 지정하지 않는다(GPU 미할당). K=11 실루엣 탐색(K=6~16)이 1분 38초
# 걸렸으므로 고정 K 하나는 그보다 빠를 것 — --time 1시간은 넉넉히.
#
# 완료 후 확인: data/cluster_centroids_uni2native_k5.pt 생성 여부.
# 이 뒤에 sbatch/pma_clusterpool_pdaccons_k5_2seed_kfold_array_hpc.sh를 제출할 것(centroids
# 파일이 있어야 함 — 없으면 ViT_PMA가 FileNotFoundError로 즉시 실패).
#
# 제출: sbatch sbatch/fit_clusters_uni2native_k5_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

echo "=== PAAD uni2native cluster fit (K=5) Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m data.fit_clusters_uni2native --k 5 --max-patches-per-slide 3000 \
    --out-path data/cluster_centroids_uni2native_k5.pt
echo "=== PAAD uni2native cluster fit (K=5) Complete: $(date) ==="
