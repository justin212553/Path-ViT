#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=standard       
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7        
#SBATCH --mem=64G                  
#SBATCH --time=24:00:00            
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/preprocess_progress.log

cd /pub/wonseukl/Path-ViT/
source .venv/bin/activate

# 파이썬 실행
./.venv/bin/python -u ./data/preprocess.py