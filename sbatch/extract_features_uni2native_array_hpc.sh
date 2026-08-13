#!/bin/bash
#SBATCH --job-name=PVT-extract-uni2native-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/extract_uni2native_array_%a.log

# preprocess_uni2native_retile_array_hpc.sh(256px@0.5MPP 재타일링, data/patches_{tcga,cptac}_uni2native/)
# 완료 후, 그 타일에 UNI2-h feature를 추출한다. --backbone uni2native가
# UNI2_NATIVE_PATCH_TRANSFORM(리사이즈 없음, 이미 256px)을 쓴다(utils/extract_features.py
# BACKBONE_REGISTRY 참조) — 기존 "uni2"(1024px 원본을 512로 리사이즈)와 다른 항목.
#
# 결과: data/patches_{tcga,cptac}_uni2native/tiles/<slide_id>/features_uni2.pt
#   (별도 디렉토리 트리라 out_filename은 그냥 기본 features_uni2.pt — 기존 산출물과 안 겹침)
#
# SLURM_ARRAY_TASK_ID: 0=tcga, 1=cptac.
#
# 완료 후: scripts/reconcile_uni2native_features.py로 기존 patches 트리에
#   features_uni2native.pt/coords_uni2native.pt로 복사(로컬 다운로드는 이 작은 결과물만).
#
# 제출: sbatch sbatch/extract_features_uni2native_array_hpc.sh
# (재타일링 16개 shard가 전부 끝난 뒤 제출할 것)

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS=(tcga cptac)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "=== extract_features(uni2native) dataset=${DATASET} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m utils.extract_features --dataset "${DATASET}" --backbone uni2native \
    --patches-root "data/patches_${DATASET}_uni2native"
echo "=== extract_features(uni2native) dataset=${DATASET} Complete: $(date) ==="
