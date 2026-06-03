#!/bin/bash
#SBATCH --job-name=path_vit_eval
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                  
#SBATCH --time=18:00:00            

cd /pub/wonseukl/Path-ViT/

conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u ./eval.py