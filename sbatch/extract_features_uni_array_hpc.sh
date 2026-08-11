#!/bin/bash
#SBATCH --job-name=PVT-extract-uni-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/extract_uni_array_%a.log

# UNI(ViT-L/16, MahmoodLab, models/uni_encoder.py) 성분(feature) 사전추출.
# 2026-08-11: HPC엔 UNI2만 추출돼 있고 UNI(v1)은 없는 것으로 확인됨(scripts/
# train_pancancer_paad_brca.py — PAAD+BRCA 공동학습 실험이 BRCA 쪽 HF 데이터셋에 UNI2가
# 없어서 UNI v1으로 통일하기로 함, PAAD/CPTAC 둘 다 UNI v1으로 다시 뽑아야 이 실험이 돌아감).
# SLURM_ARRAY_TASK_ID: 0=tcga, 1=cptac — extract_features_uni2_array_hpc.sh와 동일 관례.
#
# 사전 준비(필수, 안 되면 401 GatedRepoError로 즉시 실패):
#   https://huggingface.co/MahmoodLab/UNI 접근 승인 + .env의 HF_TOKEN이 승인된 계정 토큰인지 확인
#   (UNI2 추출 때 이미 확인된 계정이면 UNI도 같은 계정으로 승인돼 있을 가능성이 높음).
#
# 소요 시간: findings_backlog.md 기록상 로컬 RTX4060(8GB)에서 TCGA+CPTAC 전체(1028장) UNI
# 추출에 약 3시간16분 — UNI2(ViT-H, patch14)보다 가벼운 모델(ViT-L/16)이라 HPC A30에서는
# UNI2 추출보다 더 빠르게 끝날 것으로 예상.
#
# 완료 후: data/patches_{tcga,cptac}/tiles/<slide_id>/features_uni.pt 로 저장됨
#   (features.pt/features_uni2.pt와 별도 파일이라 기존 산출물을 덮어쓰지 않음)
#
# 제출: sbatch sbatch/extract_features_uni_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS=(tcga cptac)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "=== extract_features(uni) dataset=${DATASET} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m utils.extract_features --dataset "${DATASET}" --backbone uni
echo "=== extract_features(uni) dataset=${DATASET} Complete: $(date) ==="
