#!/bin/bash
#SBATCH --job-name=PVT-PMA-tileaug-pilot
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_ex_ss_aux_tileaugment_pilot.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# AssocGrpBillingMinutes(그룹 할당량 소진)로 원래 스크립트(3일 x A30 x 128G x 3시드)가 거부됨.
# 요청한 --time만큼 미리 billing minutes로 잡히므로, 훨씬 작게 요청해서 통과되는지부터 확인하는
# 파일럿이다 — seed 1개, --time=4시간 제한(train.py는 --epochs 오버라이드가 없어 config.py
# 기본 30 epoch 그대로 시도하고, 4시간 벽에 걸려 중간에 잘리는 걸로 몇 epoch/시간당 얼마나
# 도는지 실측한다). 통과되면 그 실측치로 본 실행(scripts/_pma_ex_ss_aux_tileaugment_ext.sh)의
# --time/--gres를 다시 맞춰서 제출.

python -u ./train.py --dataset tcga --seed 42 --PMA --rna-genes literature_1500 \
    --patch-keep-frac 0.8 --rna-aux-weight 1.0 --image --tile-augment \
    --external --group-ts 0721pma_tileaugment_pilot \
    2>&1 | tee .logs/train_tcga_seed42_PMA_EX_SS_AUX_AUG_pilot_ext.log
