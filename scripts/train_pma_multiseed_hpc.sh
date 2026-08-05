#!/bin/bash
#SBATCH --job-name=PVT-PMA-multiseed
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_pma_multiseed.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

# 2026-08-04: PMA(TCGA->CPTAC) leakage-free 결과의 seed 강건성 확인용.
# SS+AUX+DISP, precomputed features(실시간 aug 없음 - --image/--tile-augment 미사용, train_pma_hpc.sh와
# 다른 config이므로 별도 스크립트로 분리) — 로컬에서 이미 seed42(5-fold pooled internal c=0.5647,
# external mean=0.558)를 확보해뒀다. M7은 3-seed(42/84/126) 전부 안정적이었지만, PMA는 반대 방향
# (CPTAC->TCGA)에서 seed84가 external chance 수준(0.50)으로 붕괴하는 걸 확인한 뒤라 이 방향도
# 나머지 seed 84/126을 각 5-fold(=10 run)씩 검증해서 init-seed 하나에 기댄 결론이 아닌지 확인한다.
# 로컬 GPU를 계속 점유하지 않으려고 HPC 유휴 free-gpu 파티션으로 옮겨 실행.
#
# 끝나면 .logs/kfold_preds/tcga_PMA_EXTfdr0.1_SS_AUX_DISP_FOLD*OF5_seed{84,126}_fold*of5.csv를
# 로컬로 가져와(scp/rsync) scripts/pool_kfold_preds.py --dataset tcga --model PMA_EXTfdr0.1_SS_AUX_DISP
# --seed {84,126} --n-folds 5 로 pooling.
for seed in 84 126; do
    for fold in 0 1 2 3 4; do
        echo "=== PMA TCGA seed${seed} fold ${fold} ==="
        python -u ./train.py --PMA --dataset tcga --external --seed "$seed" \
            --rna-genes literature_fdr0.1_tcga_only \
            --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
            --fold "$fold" --n-folds 5 "$@"
    done
done
