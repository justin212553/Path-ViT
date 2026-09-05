#!/bin/bash
#SBATCH --job-name=PVT-BRCA-attn-alt-test
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_attn_alt_test_fold0seed84.log

# 2026-09-05: PAAD(fold0/seed84)에서 나이스트롬 대안 3종(fix-nystrom-landmarks/knn-mean-agg/
# cluster-attn) 전부 baseline보다 나빴다(internal 0.52~0.54 vs 0.536, external 0.57~0.58 vs
# 0.630) — "표본이 너무 작아서 patch-mixing 자체가 안 되는 것"이라는 가설을, 표본이 7배인
# BRCA(fold0/seed84, consistency 882유전자 패널)에서 같은 3종으로 재확인한다.
#
# 주의 — --fix-nystrom-landmarks는 BRCA에서 사실상 무의미할 가능성이 높다: 이 결함은 슬라이드당
# 패치 수 < num_landmarks(128)일 때만 발생하는데, BRCA는 슬라이드당 패치 수 중앙값이 10,309라
# 항상 128을 훨씬 초과한다(PAAD는 중앙값 67 < 128이라 결함이 실제로 발생). 그래도 비교 완결성을
# 위해 포함 — "역시 변화 없음"이 나오면 그 자체로 확인.
#
# --cluster-attn이 가장 중요한 검증 대상 — O(N*K)라 이론상 BRCA 규모(최대 67,268 패치)에서도
# RelativeBiasFullAttention(dense O(N^2), BRCA에서 이미 OOM 확인됨, config.py 주석)처럼 터지지
# 않아야 한다. 여기서 OOM나면 그 가설도 기각.
#
# 비교 기준(오늘 채택한 BRCA baseline, consistency 882유전자+staging, 2seed x 5fold pooled):
#   M4/PMA: internal 0.7260 [0.6774, 0.7694]
#   (이번은 단일 fold0/seed84라 이 pooled 수치와 직접 비교는 안 되고, 같은 fold를 baseline
#   없이(attention 플래그 없이) 한 번 더 따로 돌려서 "이번 fold에서의" 기준선도 같이 만든다.)
#
# 제출: sbatch sbatch/brca_attn_alternatives_test_fold0seed84_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

BASE_ARGS="--seed 84 --fold 0 --n-folds 5 --gene-selection consistency --clinical-staging --external-tss none"

echo "=== 0) baseline(attention 플래그 없음, 이번 fold 기준선) Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 $BASE_ARGS --group-ts 0905_brca_attn_alt_test 2>&1 | tee .logs/brca_attn_test0_baseline_fold0seed84.log
echo "=== 0) Complete: $(date) ==="

echo "=== 1) Nystrom landmark 버그 수정 (--fix-nystrom-landmarks, BRCA에선 사실상 no-op 예상) Start: $(date) ==="
python -u -m scripts.train_brca_m4 $BASE_ARGS --fix-nystrom-landmarks --group-ts 0905_brca_attn_alt_test 2>&1 | tee .logs/brca_attn_test1_nystromfix_fold0seed84.log
echo "=== 1) Complete: $(date) ==="

echo "=== 2) kNN 평균 집계, attention 없음 (--knn-mean-agg) Start: $(date) ==="
python -u -m scripts.train_brca_m4 $BASE_ARGS --knn-mean-agg --group-ts 0905_brca_attn_alt_test 2>&1 | tee .logs/brca_attn_test2_knnmeanagg_fold0seed84.log
echo "=== 2) Complete: $(date) ==="

echo "=== 3) 클러스터 압축 + 슈퍼토큰 dense attention (--cluster-attn) Start: $(date) ==="
python -u -m scripts.train_brca_m4 $BASE_ARGS --cluster-attn --n-clusters 16 --group-ts 0905_brca_attn_alt_test 2>&1 | tee .logs/brca_attn_test3_clusterattn_fold0seed84.log
echo "=== 3) Complete: $(date) ==="
