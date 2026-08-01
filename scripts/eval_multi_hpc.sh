#!/bin/bash
#SBATCH --job-name=PVT-EVAL-tcga
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/eval_multi.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# train_multi.py 학습이 quota 소진 등으로 마지막 internal/external 평가를 못 돌린 경우,
# 다운로드해둔 체크포인트(models/checkpoint/survival_tcga_M1M2M3PMA_{name}_best_multi.pt)만으로
# 재학습 없이 평가만 재현한다. 재학습이 아니라 forward-only라 --mem은 train_multi.py용보다
# 낮게 잡았지만, --time은 오히려 넉넉하게 잡았다 — 이 스크립트는 train_multi.py와 달리
# backbone forward를 모델 4개가 공유하지 않고 각자 독립적으로 돌리는 데다, 평가 대상도
# train+test+external(약 280명, 학습 1epoch의 train만 91명보다 훨씬 많음)이라 실측 전엔
# 정확한 소요 시간을 알 수 없다(free-gpu는 SU 비용이 0이라 넉넉히 잡아도 손해 없음).
python -u ./eval_multi.py --dataset tcga --external --model-tag M1M2M3PMA --M1 --M2 --M3 --PMA --rna-aux-weight 1.0
