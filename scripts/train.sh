#!/bin/bash
#SBATCH --job-name=path_vit_train
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8          # num_workers=8 에 맞춤 (DataLoader worker + 예비)
#SBATCH --mem=64G                  # 대형 WSI 로딩 버퍼 (패치 수천 장 × patient 단위)
#SBATCH --time=18:00:00            # ViT 수렴을 위해 24시간 연속 가동 요새 방어
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_progress.log

cd /pub/wonseukl/Path-ViT/

source .venv/bin/activate

./.venv/bin/python -u ./train.py