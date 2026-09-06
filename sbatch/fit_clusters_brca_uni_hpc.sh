#!/bin/bash
#SBATCH --job-name=PVT-fit-clusters-brca
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/fit_clusters_brca_uni_%j.log

# 2026-09-05: BRCA에 클러스터풀(models/vit_pma.py cluster_pool)을 이식하기 위한 1단계 —
# data/fit_clusters_uni2native.py와 동일 원리를 BRCA(backbone="uni", UNI v1 1024차원, PAAD의
# uni2native/uni2와는 다른 feature 공간이라 그쪽 centroids를 재사용 불가)에 맞춰 적합한다.
# BRCA feature(data/patches_tcga_brca/tiles/{slide_id}/features_uni.pt)는 로컬엔 없고 HPC에만
# 있다(brca_for_hpc.zip으로 이미 전송됨) — 그래서 이 단계 자체를 HPC에서 돌려야 한다.
#
# k-means 자체는 GPU가 필요 없는 CPU 연산(sklearn MiniBatchKMeans)이지만, free-gpu 파티션이
# 이 클러스터에서 가장 대기시간이 짧아 그대로 사용(--gres 지정 안 함 = GPU 미할당, CPU만 씀).
#
# PAAD(TCGA-only, 203슬라이드/547K패치)에서 K=6~16 실루엣 탐색이 1분 38초 걸렸다 — BRCA는
# 1058케이스/약 1131슬라이드로 데이터량이 ~5~6배라 10~15분 정도 예상. --time 2시간은 넉넉히.
#
# 완료 후 확인: data/cluster_centroids_brca_uni.pt 생성 여부, 로그의 "최적 K" 값.
# 이 뒤에 sbatch/brca_m4_clusterpool_kfold_array_hpc.sh를 제출할 것(centroids 파일이 있어야
# 함 — 없으면 ViT_PMA가 FileNotFoundError로 즉시 실패).
#
# 제출: sbatch sbatch/fit_clusters_brca_uni_hpc.sh
# (완료 확인 후) sbatch sbatch/brca_m4_clusterpool_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

echo "=== BRCA uni cluster fit Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m data.fit_clusters_brca_uni --eval-k 6 16 --max-patches-per-slide 3000
echo "=== BRCA uni cluster fit Complete: $(date) ==="
