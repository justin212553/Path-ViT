#!/bin/bash
#SBATCH --job-name=path_vit_train
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8          # num_workers=8 에 맞춤 (DataLoader worker + 예비)
#SBATCH --mem=32G                  
#SBATCH --time=18:00:00            
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_progress.log

cd /pub/wonseukl/Path-ViT/

source .venv/bin/activate

./.venv/bin/python -u ./train.py