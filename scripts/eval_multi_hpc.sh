#!/bin/bash
#SBATCH --job-name=PVT-EVAL-tcga
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/eval_multi.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# train_multi.py 학습이 quota 소진 등으로 마지막 internal/external 평가를 못 돌린 경우,
# 다운로드해둔 체크포인트(models/checkpoint/survival_tcga_M1M2M3PMA_{name}_best_multi.pt)만으로
# 재학습 없이 평가만 재현한다. 재학습이 아니라 forward-only라 --mem/--time을 train_multi.py용
# 스크립트보다 훨씬 낮게 잡았다.
python -u ./eval_multi.py --dataset tcga --external --model-tag M1M2M3PMA --M1 --M2 --M3 --PMA
