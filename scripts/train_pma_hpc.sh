#!/bin/bash
#SBATCH --job-name=PVT-PMA-tcga
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_pma.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# train_m1/m2/m3_hpc.sh와 동일한 이유(컴퓨트 노드 인터넷 문제로 wandb.log()가 멈추는 것 대응)
export WANDB_MODE=offline

# 2026-08-04: --rna-genes literature_fdr0.1_tcga_only — data/select_rnaseq_genes.py
# --single-cohort tcga 산출물(CPTAC train 라벨을 전혀 안 본 leakage-free 유전자 리스트).
# 기존 both-결합 리스트(literature_1500)는 external로 쓰는 반대 코호트 case가 유전자 선정
# 단계에서 이미 노출돼 있었다(실측 약 60% 중복) — M6/M7/PMA 전부 leakage 제거 후 재검증 중.
# PMA(WSI+RNA+Clinical, --no-clinical 없음)만 단독 학습 — train_m1/m2/m3_hpc.sh와 동일한
# 이유로 재현성 격리를 위해 단독 실행. SS+AUX+AUG+DISP 전부 적용(RNA 있으니 AUX 포함).
#
# --tile-decode-workers 8 / --cache-val-tiles / --cache-external-tiles: train_m1_hpc.sh와
# 동일한 이유(타일 디코딩 스레드풀 하드코딩 + evaluate()가 val/external을 캐시 없이 디스크에서
# 매 epoch 새로 디코딩하던 게 진짜 병목이었음).
python -u ./train.py --PMA --dataset tcga --external --seed 42 \
    --rna-genes literature_fdr0.1_tcga_only \
    --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles "$@"
