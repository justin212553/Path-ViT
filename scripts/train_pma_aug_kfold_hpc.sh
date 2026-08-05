#!/bin/bash
#SBATCH --job-name=PVT-PMA-aug-kfold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_pma_aug_kfold.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

# 2026-08-04: leaky 실시간-aug PMA(단일 split, literature_1500)가 internal c=0.6352(+3.67σ),
# external c=0.6517(+5.29σ)로 지금까지 본 leakage-free PMA(1.6~2.9σ)보다 훨씬 강한 신호를 보였다
# (null c-index는 무작위 risk-score 500회 시뮬레이션으로 측정: TCGA mean=0.4995 std=0.037,
# CPTAC mean=0.4984 std=0.029). 이게 진짜 aug 효과인지, 아니면 leakage+단일split 노이즈인지
# 가리려면 leakage-free(fdr0.1_tcga_only) + k-fold로 재검증해야 한다.
#
# 5개 fold를 순차 루프 대신 "같은 GPU에 5개 프로세스 동시 실행"으로 띄운다 - fold마다 완전히
# 독립된 프로세스라 train_multi.py에서 겪었던 RNG 공유 문제(한 프로세스 안에서 모델을 순차로
# 만들 때 생기던 초기화 스트림 오염)는 해당 없음. real-time aug는 매 epoch CNN 인코딩을 새로
# 하므로 순차 루프보다 훨씬 오래 걸려서 병렬화 이득이 큼. A30(24GB) 기준으로는 5개 동시 실행이
# 들어갈 것으로 예상 - OOM 나면 아래 fold 목록을 두 번에 나눠 돌릴 것(예: "0 1 2"와 "3 4").
# --tile-decode-workers는 프로세스당 4(위 --cpus-per-task=20을 5개 프로세스가 나눠 쓰는 셈,
# 기존 단일 실행 스크립트의 8보다 낮춤 - 5개가 동시에 8씩 쓰면 CPU 40개를 요구해 오히려
# 컨텍스트 스위칭으로 느려진다).
for fold in 0 1 2 3 4; do
    python -u ./train.py --PMA --dataset tcga --external --seed 42 \
        --rna-genes literature_fdr0.1_tcga_only \
        --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
        --tile-decode-workers 4 --cache-val-tiles --cache-external-tiles \
        --fold "$fold" --n-folds 5 "$@" \
        > ".logs/pma_aug_kfold_fold${fold}.log" 2>&1 &
done
wait
echo "=== 5개 fold 전부 종료 ==="
