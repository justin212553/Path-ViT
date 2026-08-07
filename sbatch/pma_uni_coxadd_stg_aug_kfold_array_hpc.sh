#!/bin/bash
#SBATCH --job-name=PVT-PMA-uni-coxadd-stg-aug-kfold-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_uni_coxadd_stg_aug_kfold_array_%a.log

# 로컬 no-aug 파일럿(PMA_uni_INT1500_SS_AUX_STG_R_DISP_COX_ADD, seed84)에서 확인한 것:
#   internal(pooled) 0.617 / external(mean) 0.644 — 처음으로 M3(WSI+RNA, clinical 없음, UNI,
#   external 0.638)를 external에서 넘었다. 다만 bootstrap 검정 결과 이 격차(external-internal)의
#   95% CI가 0을 포함해([-0.057, +0.116]) 통계적으로 확정적이진 않다.
# 그 뒤 확인한 것: M3(ResNet50)에서 AUG는 평균 성능은 거의 안 바꾸면서(0.633->0.628) fold 간
# internal 분산을 30% 줄였다(std 0.063->0.045) — "값의 크기"가 아니라 "값의 안정성"에 기여하는
# 걸로 보인다.
#
# 이번엔 이 둘을 합친다: PMA(WSI+RNA+Clinical) + UNI + cox_add(margin+staging) + real-time AUG
# + --strong-blur(2026-08-07 신규 — GaussianBlur만 kernel_size 3->5, sigma 상한 1.0->2.0,
# 적용확률 0.15->0.35로 세게. ColorJitter/flip은 그대로 둬서 염색강도 정보는 안 건드림).
# AUG가 UNI+cox_add+staging 조합에도 "분산을 줄이는" 효과를 내는지, 그래서 internal<external
# 격차의 신뢰구간이 좁아지는지가 이번 실험의 핵심 질문.
#
# --time=24h: M3+UNI(no-aug)는 fold당 수 분 내로 끝났지만, 이 스크립트는 real-time
# augmentation(--tile-augment --image, 매 epoch 타일 재디코딩+UNI forward)이 붙어 있어
# 시간이 훨씬 오래 걸린다 — ResNet50+AUG가 fold당 실측 3.2~3.7시간이었는데 UNI 백본은 forward
# 연산량이 훨씬 커서 이보다 상당히 오래 걸릴 수 있다. 제출 후 첫 fold의 epoch 1개 소요 시간을
# 보고 24h 안에 들어올지 가늠해볼 것(안 되면 --epochs를 줄이거나 --time을 늘려 재제출).
#
# 완료 후 집계(model_prefix에 _STG_R..._AUG_BLUR..._COX_ADD 태그가 다 들어간다는 점 주의):
#   python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model PMA_uni_INT1500_SS_AUX_STG_R_DISP_AUG_BLUR_COX_ADD
#
# 제출: sbatch sbatch/pma_uni_coxadd_stg_aug_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

FOLD=$SLURM_ARRAY_TASK_ID
log=".logs/train_tcga_seed84_PMA_uni_INT1500_SS_AUX_STG_R_DISP_AUG_BLUR_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== PMA(uni,cox_add,STG+R,AUG+BLUR) fold=${FOLD}/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed 84 \
    --backbone uni \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --tile-augment --image --strong-blur --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
    --fold "${FOLD}" --n-folds 5 --group-ts 0807pma_uni_coxadd_stg_aug_blur_kfold5_array_seed84 2>&1 | tee "${log}"
echo "=== PMA(uni,cox_add,STG+R,AUG+BLUR) fold=${FOLD}/5 Complete: $(date) ==="
