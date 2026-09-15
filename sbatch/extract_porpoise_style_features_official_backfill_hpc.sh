#!/bin/bash
#SBATCH --job-name=PVT-porpoise-feat-backfill
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/extract_porpoise_style_features_official_backfill_%j.log

# 2026-09-15: PORPOISE 공식 mutsig 166명 목록 중 이 연구 자체 185명 코호트 기준 추출에서
# 빠졌던 35명을 추가로 채운다. uni2native 타일은 이미 있고(data/patches_tcga_uni2native/
# tiles/), 이미 존재하는 .pt는 건너뛰므로(data/extract_porpoise_style_features.py 자체
# 로직) 이 35명분만 새로 GPU forward pass를 태운다 — 몇 분 안에 끝날 가벼운 작업이라
# 짧은 시간/자원으로 잡았다.
#
# 제출: sbatch sbatch/extract_porpoise_style_features_official_backfill_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

echo "=== PORPOISE 공식 mutsig 목록 기준 ResNet50 feature backfill Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m data.extract_porpoise_style_features \
    --csv-path porpoise/datasets_csv_mutsig/tcga_paad_all_clean.csv.zip
echo "=== PORPOISE 공식 mutsig 목록 기준 ResNet50 feature backfill Complete: $(date) ==="
