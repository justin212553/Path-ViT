#!/bin/bash
#SBATCH --job-name=PVT-porpoise-feat
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/extract_porpoise_style_features_%j.log

# 2026-09-05: sota/PORPOISE/inputs/tcga_paad_20x_features/pt_files/의 기존 .pt가 PORPOISE
# 원본이 아니라 우리 UNI2 feature(1536차원)를 재포장한 것이었음을 확인(진짜 원본은 ImageNet
# ResNet50 truncated, 1024차원) — "PORPOISE가 우리와 다른 전처리로 신호를 뽑아낸 게 아닐까"를
# 직접 검증하기 위해 진짜 원본 스펙으로 재추출한다.
#
# 패치 위치는 새로 tissue segmentation 안 하고 이미 검증된 UNI2-h 공식 feature의 coords를
# 재사용(data/uni2h_official_features/{tcga,cptac}/*.h5) — "같은 패치, 다른 backbone"으로
# 통제해야 backbone 차이만 순수 비교 가능. WSI 원본(.svs) 필요 — HPC에 있어야 함(data/
# tcga_paad_wsi/, data/cptac_pda_wsi/).
#
# 완료 후: data/porpoise_style_features/{tcga,cptac}/pt_files/*.pt (N x 1024, ImageNet
# ResNet50 layer3 truncated) 생성 확인.
#
# 제출: sbatch sbatch/extract_porpoise_style_features_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

echo "=== PORPOISE-style(ImageNet ResNet50 truncated, 1024-dim) feature 추출 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m data.extract_porpoise_style_features --datasets tcga,cptac
echo "=== PORPOISE-style feature 추출 Complete: $(date) ==="
