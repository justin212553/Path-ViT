#!/bin/bash
#SBATCH --job-name=PVT-PMA-resnet50-coxadd-stg-aug-kfold-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_resnet50_coxadd_stg_aug_kfold_array_%a.log

# 2026-08-08: backbone(UNI vs ResNet50) 자체는 거의 영향이 없고(margin만 있는 레시피로 통제
# 비교, external 0.616 vs 0.621 — scripts/compare_pma_backbone_ext.py), 지금까지 최고 기록
# (0.644)의 진짜 원인은 staging 추가였다는 게 오늘 확인됐다. 그런데 AUG는 UNI에서는 계속
# 손해였던 반면(0.644->0.614/0.602) ResNet50에서는 과거에 손해가 아니었다(M3 기준 평균은
# 거의 그대로, fold 간 internal 분산만 30% 줄어드는 안정화 효과) — "UNI/ResNet50이 거기서
# 거기"라면, AUG가 안 깨지는 ResNet50 쪽에 staging까지 얹으면 오히려 UNI 없이도 0.644에
# 근접하거나 넘어설 수 있는지 확인한다. 로컬에서 이미 도는 중인 no-aug 버전
# (PMA_INT1500_SS_AUX_STG_R_DISP_COX_ADD, backbone 기본값=resnet50)과 나란히 비교할 aug 버전.
#
# ResNet50+AUG(real-time, --tile-augment --image) fold당 실측 시간 참고: M3(ResNet50) 기준
# 과거 ~3.2~3.7시간/fold였다 — UNI보다 훨씬 가벼워 24h 안에 5-fold 전부 여유 있게 들어올 것으로
# 예상.
#
# 완료 후 집계(model_prefix에 _STG_R_AUG_DISP_COX_ADD 태그, backbone 기본값이라 _uni 태그 없음):
#   python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model PMA_INT1500_SS_AUX_STG_R_AUG_DISP_COX_ADD
#
# 제출: sbatch sbatch/pma_resnet50_coxadd_stg_aug_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

FOLD=$SLURM_ARRAY_TASK_ID
log=".logs/train_tcga_seed84_PMA_INT1500_SS_AUX_STG_R_AUG_DISP_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== PMA(resnet50,cox_add,STG+R,AUG) fold=${FOLD}/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed 84 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
    --fold "${FOLD}" --n-folds 5 --group-ts 0808pma_resnet50_coxadd_stg_aug_kfold5_array_seed84 2>&1 | tee "${log}"
echo "=== PMA(resnet50,cox_add,STG+R,AUG) fold=${FOLD}/5 Complete: $(date) ==="
