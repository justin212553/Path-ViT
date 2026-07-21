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
# 91명(TCGA train) 처리에 30분 이상, epochs=30 기준 런 1개당 20시간 안팎 예상(A30에서는 더
# 빠를 수 있음). external은 이 세션의 표준 관례대로 tcga train -> cptac test 단일 방향, 3시드.
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
        --external --group-ts "${GroupTs}" 2>&1 | tee "${log}"
    echo "=== PMA_EX_SS_AUX_AUG seed=${seed} Complete: $(date) ==="
done

echo "=== ALL PMA_EX_SS_AUX_AUG EXTERNAL RUNS COMPLETE: $(date) ==="
