#!/bin/bash
#SBATCH --job-name=PVT-extract-uni2-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/extract_uni2_array_%a.log

# UNI2-h(ViT-H/14, MahmoodLab, models/uni2_encoder.py) 성분(feature) 사전추출.
# SLURM_ARRAY_TASK_ID: 0=tcga, 1=cptac — 코호트별로 독립 GPU 1장씩(비어있으면 동시에, 아니면
# 순서대로), m1/m2/m3/pma kfold array 스크립트와 동일한 관례.
#
# 사전 준비(필수, 둘 다 안 되면 401 GatedRepoError로 즉시 실패):
#   1) https://huggingface.co/MahmoodLab/UNI2-h 에서 접근 승인 신청 — UNI(MahmoodLab/UNI)
#      승인과는 별개로 UNI2-h용을 새로 받아야 한다(로컬 스모크 테스트로 실제 확인함, 2026-08-07).
#   2) .env의 HF_TOKEN이 승인된 계정의 토큰인지 확인(UNI 때 쓰던 토큰이 같은 계정이면 재사용 가능).
#
# 소요 시간 추정: 로컬 RTX4060(8GB)에서 UNI(ViT-L/16, 512px 입력)로 TCGA+CPTAC 전체(1028장)
# 추출에 약 3시간16분이 걸렸다. UNI2-h는 파라미터 2배(ViT-H) + patch14라 토큰 수도 더 많아
# (512px 기준 ~1024 -> ~1296) 연산량 자체는 3~4배 무겁지만, HPC A30(24GB)이 로컬 4060(8GB)보다
# 훨씬 빠르고 배치도 더 크게 잡을 여지가 있어 실제 wall time은 가늠하기 어렵다 —
# --time=24h로 안전 마진을 크게 잡아뒀다. 제출 후 첫 로그에서 노드 처리 속도(it/s)를 보고
# 필요시 utils/extract_features.py의 BACKBONE_REGISTRY["uni2"]["batch_size"](현재 8, 보수적
# 기본값)를 조정할 것.
#
# 완료 후: data/patches_{tcga,cptac}/tiles/<slide_id>/features_uni2.pt 로 저장됨
#   (features.pt/features_uni.pt와 별도 파일이라 기존 산출물을 덮어쓰지 않음 — 롤백 가능)
#
# 제출: sbatch sbatch/extract_features_uni2_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS=(tcga cptac)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "=== extract_features(uni2) dataset=${DATASET} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m utils.extract_features --dataset "${DATASET}" --backbone uni2
echo "=== extract_features(uni2) dataset=${DATASET} Complete: $(date) ==="
