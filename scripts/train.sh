#!/bin/bash
#SBATCH --job-name=path_vit_train
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                  
#SBATCH --time=18:00:00            
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_progress.log

cd /pub/wonseukl/Path-ViT/

source .venv/bin/activate

./.venv/bin/python -u ./train.py