#!/bin/bash
#SBATCH --job-name=PVT-attn-heatmaps-brca
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/generate_attention_heatmaps_brca.log

# scripts/generate_attention_heatmaps.py(BRCA 사양, 2026-08-31 재작성) — M4/PMA의 ABMIL patch
# attention + co-attention 4-component 가중치를 시각화한다. PAAD/uni2native 버전은 HPC에도
# 원본 재타일링 JPG가 없어서(features_uni2native.pt/coords_uni2native.pt만 있음, 확인됨) 실행
# 불가였다 — TCGA-BRCA(scripts/train_brca_m4.py 체크포인트, sbatch/train_brca_m4_internal_hpc.sh)
# 로 대상을 바꿨다. BRCA도 원본 patch 이미지가 없어(coords.pt+features_uni.pt뿐) 조직 사진 위
# 오버레이가 아니라 좌표 산점도(scatter) 형태의 순수 attention heatmap이다.
#
# 필요 조건: sbatch/train_brca_m4_internal_hpc.sh(seed84)가 먼저 완료돼 있어야 함
# (models/checkpoint/survival_brca_best_brca_pma_top1500_ss_aux_seed84.pt 필요).
#
# 로컬에도 BRCA feature/좌표(data/patches_tcga_brca/)는 이미 있어서, 체크포인트(.pt) 하나만
# HPC에서 받아오면 이 스크립트는 로컬에서도 그대로 돌아간다(WSI 원본이 아니라 사전추출
# feature+좌표뿐이라 용량이 checkpoint 하나 수준). 굳이 HPC까지 갈 필요 없으면 로컬 실행도 고려.
#
# 결과: .logs/attention_heatmaps/<case_id>_baseline.png (환자당 1장)
#
# 제출: sbatch sbatch/generate_attention_heatmaps_brca_hpc.sh

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== generate_attention_heatmaps(BRCA) Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.generate_attention_heatmaps --seed 84
echo "=== generate_attention_heatmaps(BRCA) Complete: $(date) ==="
