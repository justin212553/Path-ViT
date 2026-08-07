#!/bin/bash
#SBATCH --job-name=PVT-M3-uni-aug-kfold-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m3_uni_aug_kfold_array_%a.log

# m3_kfold_array_hpc.sh(ResNet50-SwAV, AUG, seed84)에서 --backbone uni만 바꾼 버전.
#
# 배경: seed84 no-aug 파일럿에서 UNI(Mahmood Lab ViT-L/16 foundation model, 병리 특화
# self-supervised 사전학습)가 ResNet50-SwAV보다 external을 크게 앞질렀다(internal 0.601/
# external 0.638±0.010 vs ResNet50 no-aug 0.626/0.567±0.017) — M6(RNA only, external 0.610)와
# 지금까지의 M3 최고 기록(ResNet50+AUG, 0.617)을 aug 없이도 넘어선 유일한 조합. AUG까지
# 결합하면 더 나아지는지 확인한다.
#
# 2026-08-06: UNI는 CNNEncoder(ResNet50)보다 훨씬 무거운 ViT-L/16이라, real-time augmentation
# (--tile-augment --image, 매 epoch 원본 타일을 디코딩+증강+forward)의 fold당 소요 시간이 얼마나
# 늘어날지 실측이 없다. backbone은 train.py가 무조건 requires_grad_(False)로 얼리므로(1660행
# 근방) backward 비용/메모리 폭증 걱정은 없지만(활성화 저장 없이 forward만), forward 자체의
# FLOPs가 ResNet50보다 훨씬 크다 — ResNet50+AUG가 fold당 실측 ~3.2~3.7시간이었으니 UNI는 이보다
# 상당히 더 걸릴 수 있다. --time=24h로 맞춰뒀지만 초과할 가능성을 배제 못 한다 — 제출 후 첫
# fold의 epoch 1개 소요 시간을 보고 24h 안에 30 epoch이 들어올지 가늠해볼 것(안 되면 --epochs를
# 줄이거나 --time을 더 늘려 재제출).
#
# utils/extract_features_augmented.py(1회성 augmented feature 사전추출, "저렴한" aug 경로)는
# CNNEncoder 전용으로 하드코딩돼 있어 UNI를 지원하지 않는다 — 그래서 이 스크립트는 유일하게
# 가능한 경로인 real-time(--image) 방식을 쓴다.
#
# 완료 후 집계(model_prefix에 _uni, --no-clinical의 _NOCLINICAL 태그가 들어간다는 점 주의):
#   python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model PMA_uni_INT1500_SS_AUX_AUG_NOCLINICAL_DISP
#
# 제출: sbatch sbatch/m3_uni_aug_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

FOLD=$SLURM_ARRAY_TASK_ID
log=".logs/train_tcga_seed84_PMA_uni_INT1500_SS_AUX_AUG_NOCLINICAL_DISP_kfold5_fold${FOLD}.log"

echo "=== M3(PMA_uni_INT1500_SS_AUX_AUG_NOCLINICAL_DISP) fold=${FOLD}/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --no-clinical --rna-genes literature_1500_intersection --dataset tcga --external --seed 84 \
    --backbone uni \
    --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
    --fold "${FOLD}" --n-folds 5 --group-ts 0806m3_uni_int1500_aug_kfold5_array_seed84 2>&1 | tee "${log}"
echo "=== M3(PMA_uni_INT1500_SS_AUX_AUG_NOCLINICAL_DISP) fold=${FOLD}/5 Complete: $(date) ==="
