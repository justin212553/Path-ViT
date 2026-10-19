#!/bin/bash
#SBATCH --job-name=PVT-ksweep-brca
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/ksweep_brca_uni_%j.log

# 2026-10-18: Methods "K=11 선택 근거"용 BRCA silhouette sweep 재실행. 2026-09-05 원래 sweep
# (sbatch/fit_clusters_brca_uni_hpc.sh)의 로그가 HPC 로그 정리로 사라져 BRCA 쪽 근거가 없다.
# 조건은 PAAD sweep·원래 BRCA sweep과 동일: 슬라이드당 최대 3,000패치, K=6~16, k-means seed 42,
# silhouette은 50,000패치 표본 중 10,000개로 계산.
#
# CPU만 쓰는 작업이라 free(CPU) 파티션. 원래 BRCA sweep은 free-gpu에서 10~15분 예상이었다 —
# --time 4시간은 대기·preemption 재시작까지 넉넉히.
#
# 결과 centroid는 기존 data/cluster_centroids_brca_uni.pt를 덮어쓰지 않도록 따로 저장한다
# (본 실험의 fold-safe 학습은 fold별 centroid를 따로 적합하므로 이 파일은 쓰지 않는다).
#
# 제출: sbatch sbatch/ksweep_brca_uni_cpu_hpc.sh
# 결과 확인: grep -E "K= *[0-9]+ +silhouette|최적 K" .logs/ksweep_brca_uni_*.log
# 선행: data/fit_clusters_brca_uni.py(2026-10-18 메모리 수정판)를 HPC에 올릴 것

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "=== BRCA K sweep (CPU) Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m data.fit_clusters_brca_uni --eval-k 6 16 --max-patches-per-slide 3000 --seed 42 \
    --out-path data/cluster_centroids_brca_uni_ksweep_20261018.pt
echo "=== BRCA K sweep (CPU) Complete: $(date) ==="
