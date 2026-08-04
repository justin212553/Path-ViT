#!/bin/bash
#SBATCH --job-name=PVT-PMA-tileaug
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_ex_ss_aux_tileaugment_ext.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology) M4_Train.ipynb처럼 학습 시 타일에
# 실시간 augmentation(RandomFlip/ColorJitter/GaussianBlur)을 적용해 매 forward마다 frozen
# backbone을 다시 태운다(--image --tile-augment, data/patch_utils.py::PATCH_TRANSFORM_AUGMENTED).
# 우리가 지금까지 써온 features.pt 사전추출 캐시(augmentation 구조적으로 불가능)와 다른 경로다.
#
# 매우 느리다 — 로컬 ResNet50/1024px 실측 24 img/s 기준 slide당 최대 512타일 x epoch당 약
# 91명(TCGA train) 처리에 30분 이상, epochs=30 기준 런 1개당 20시간 안팎 예상.
#
# 2026-08-03: 24 img/s는 --tile-decode-workers 기본값(4)으로 측정된 수치다 — 실제 병목은 GPU가
# 아니라 models/vit_m1.py::_patch_tokens의 타일 디코딩+증강 스레드풀(그동안 4로 하드코딩, 이제
# CLI로 노출)이었다. 이 작업은 DataLoader(num_workers)가 아니라 forward() 안에서 도는 별도
# 스레드풀이라, 아래 --cpus-per-task=8을 실제로 다 쓰려면 --tile-decode-workers도 맞춰줘야 한다
# (안 그러면 8 CPU를 예약해놓고 4개만 쓰는 셈). A30 자체는 FP32/메모리대역폭 기준 RTX 4060보다
# 압도적으로 빠르지 않다 — 진짜 절감은 이 스레드 수를 맞추는 데서 나올 가능성이 크다.
#
# 제출: sbatch scripts/_pma_ex_ss_aux_tileaugment_ext.sh

LogDir=".logs"
Seeds=(42 84 126)
GroupTs="0721pma_tileaugment_ext"

for seed in "${Seeds[@]}"; do
    echo "=== PMA_EX_SS_AUX_AUG seed=${seed} Start: $(date) ==="
    log="${LogDir}/train_tcga_seed${seed}_PMA_EX_SS_AUX_AUG_ext.log"
    python -u ./train.py --dataset tcga --seed "${seed}" --PMA --rna-genes literature_1500 \
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 --image --tile-augment \
        --tile-decode-workers 8 \
        --external --group-ts "${GroupTs}" 2>&1 | tee "${log}"
    echo "=== PMA_EX_SS_AUX_AUG seed=${seed} Complete: $(date) ==="
done

echo "=== ALL PMA_EX_SS_AUX_AUG EXTERNAL RUNS COMPLETE: $(date) ==="
