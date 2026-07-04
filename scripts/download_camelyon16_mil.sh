#!/bin/bash
#SBATCH --job-name=camelyon16_mil_download
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/camelyon16_mil_download_%j.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

# 기본: resnet50_bt (지금 backbone과 같은 Kang et al. 2023 벤치마크 계열)
# 다른 feature로 받으려면: sbatch scripts/download_camelyon16_mil.sh --features UNI
python -u -m utils.download_camelyon16_mil --data-root data/patches "$@"
