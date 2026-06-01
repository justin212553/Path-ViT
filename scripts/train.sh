#!/bin/bash
#SBATCH --job-name=path_vit_train
#SBATCH --partition=gpu            
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1                   # GPU 1대 독점 징집 (A30, V100 등 배정)
#SBATCH --cpus-per-task=4          # 데이터 로딩(DataLoader num_workers) 가속용 CPU 4코어 배정
#SBATCH --mem=32G                  # 배치 이미지 버퍼 적재를 위한 32GB 메모리 타격
#SBATCH --time=24:00:00            # ViT 수렴을 위해 24시간 연속 가동 요새 방어
#SBATCH --output=/pub/wonseukl/Path-ViT/.log/train_progress.log

cd /pub/wonseukl/Path-ViT/

source .venv/bin/activate

python3 -u train.py \
    --data_dir /pub/wonseukl/Path-ViT/data/patches \
    --batch_size 256 \
    --epochs 50 \
    --lr 1e-4