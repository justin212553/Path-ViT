#!/bin/bash
#SBATCH --job-name=PVT-BRCA-cp-clr100-test
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_clusterpool_clr100_fold0seed84_test.log

# 2026-09-05: BRCA cluster_pool(10-fold 앙상블 internal C=0.7320, M7 대비 delta +0.0276이나
# p=0.315로 아직 비유의)에 CNV/mutation(BRCA 데이터/코드 자체가 없음 — PDAC 전용 하드코딩,
# 포팅 필요) 대신 먼저 --clinical-lr-mult(코드는 이미 있고 데이터 불필요, 즉시 테스트 가능)만
# 얹었을 때 효과가 있는지 fold0/seed84 하나로 먼저 확인(사용자 결정: "CNV/mutation 빼고
# CLR 먼저"). PAAD에서 CLR100이 branch-competition을 해소해 M4 성능을 끌어올린 핵심 레버였던
# 것과 동일한 값(100)으로 시작.
#
# 승산이 있어 보이면 이 fold0/seed84 결과를 보고 10-fold 전체 array job으로 확장할 것
# (sbatch/brca_m4_clusterpool_kfold_array_hpc.sh를 --clinical-lr-mult 100 추가해서 복제).
#
# 제출: sbatch sbatch/brca_m4_clusterpool_clr100_fold0seed84_test_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

echo "=== BRCA cluster_pool+CLR100 seed=84 fold=0/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed 84 --fold 0 --n-folds 5 \
    --gene-selection consistency --clinical-staging --cluster-pool --clinical-lr-mult 100 \
    --external-tss none --group-ts 0905_brca_m4_clusterpool_clr100_fold0seed84_test \
    2>&1 | tee .logs/train_brca_m4_clusterpool_clr100_fold0seed84.log
echo "=== BRCA cluster_pool+CLR100 seed=84 fold=0/5 Complete: $(date) ==="
