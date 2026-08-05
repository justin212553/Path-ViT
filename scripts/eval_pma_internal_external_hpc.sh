#!/bin/bash
#SBATCH --job-name=PVT-EVAL-pma
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/eval_pma_internal_external.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# scripts/eval_pma_internal_external.py — 재학습 없이 기존 checkpoint만으로 train.py의
# 학습 종료 직후 internal/external 평가 블록(train.py:1856-1912)을 그대로 재현한다
# (eval_multi_hpc.sh와 동일한 용도, PMA 단일 checkpoint 버전). --image(raw 이미지 실시간
# 인코딩) 모드로 학습된 checkpoint라 forward마다 타일 디코딩이 들어가므로, 순수 속도용으로
# --tile-decode-workers를 --cpus-per-task=8에 맞춘다(결과에는 영향 없음).
#
# 기본 인자는 survival_tcga_seed42_EXTfdr0.1_SS_AUX_PMA_EXTfdr0.1_SS_AUX_AUG_DISP_best_pma.pt
# 학습 커맨드(--dataset tcga --seed 42 --rna-genes literature_fdr0.1_tcga_only --image
# --tile-augment --attn-dispersion --rna-aux-weight 1.0)에 맞춰져 있다 — 다른 PMA checkpoint를
# 평가하려면 --checkpoint와 함께 그 checkpoint를 학습할 때 쓴 값으로 나머지 인자도 바꿔야 한다.
#
# 제출: sbatch scripts/eval_pma_internal_external_hpc.sh
python -u -m scripts.eval_pma_internal_external --tile-decode-workers 8 "$@"
