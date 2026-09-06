#!/bin/bash
#SBATCH --job-name=PVT-porpoise-feat-cptac
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/extract_porpoise_style_features_cptac_%j.log

# 2026-09-06: TCGA-PAAD PORPOISE 재현(AMIL/MMF)이 끝나면 우리 프로젝트 관례대로 CPTAC을
# external cohort로도 평가해볼 예정이라(PORPOISE 자체엔 이 개념이 없음 — 순수 internal
# 5-fold CV뿐), CPTAC 슬라이드도 미리 같은 스펙(ImageNet ResNet50, layer3 truncated,
# 1024차원, 256px@0.5MPP)으로 추출해 둔다.
#
# data/extract_porpoise_style_features.py --dataset cptac 사용 — TCGA와 달리 PORPOISE 공식
# CSV가 CPTAC을 아예 지원하지 않으므로, 슬라이드 목록은 CSV가 아니라 uni2native 리타일링이
# 이미 처리해 둔 data/patches_cptac_uni2native/tiles/ 아래 슬라이드 폴더 전부를 그대로 쓴다.
# TCGA와 마찬가지로 SVS/openslide 접근이나 tissue segmentation 재구현 없음(이미 tissue-
# segmented jpg 재사용, GPU forward pass만 새로 함).
#
# 선행 조건: sbatch/preprocess_uni2native_retile_array_hpc.sh(CPTAC shard) 완료돼 있어야 함
# (기존 uni2native 파이프라인에서 이미 끝난 것으로 확인됨 — data/cluster_features_uni2native_
# cptac.csv 등 CPTAC uni2native 산출물이 로컬에 이미 존재).
#
# 로컬 CPTAC-PDA 원본 WSI 567개(TCGA-PAAD 466개보다 많음) — 시간 여유 있게 8h로 설정.
#
# 완료 후: data/porpoise_style_features/cptac/pt_files/*.pt 생성 확인. TCGA 쪽과 합쳐서
# external 평가 스크립트(추후 작성)에서 사용.
#
# 제출: sbatch sbatch/extract_porpoise_style_features_cptac_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

echo "=== PORPOISE-style(ImageNet ResNet50 truncated, 1024-dim, uni2native 타일 재사용) CPTAC feature 추출 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m data.extract_porpoise_style_features --dataset cptac
echo "=== PORPOISE-style CPTAC feature 추출 Complete: $(date) ==="
