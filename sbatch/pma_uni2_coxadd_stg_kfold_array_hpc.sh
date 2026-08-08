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
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_uni2_coxadd_stg_kfold_array_%a.log

# 2026-08-07(2차): --strong-blur 버전(kernel_size 5, sigma 상한 2.0, 적용확률 0.35) 결과 —
# external이 오히려 no-aug(0.644±0.011)보다 떨어졌다(0.614±0.008, internal은 반대로 0.659까지
# 오름). eval(val/test/external)은 항상 증강 없는 원본을 쓰는데 train만 blur가 세게 낀 이미지를
# 봐서, train/eval 분포 간극이 너무 벌어진 게 원인으로 추정된다 — blur를 뺀 기본 AUG(flip +
# ColorJitter + 약한 blur, kernel_size=3/sigma 상한 1.0/적용확률 0.15, PATCH_TRANSFORM_AUGMENTED_
# CACHED)로 재검증한다. --strong-blur 제거.
#
# 로컬 no-aug 파일럿(PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD, seed84) 기준값:
#   internal(pooled) 0.617 / external(mean) 0.644 ± 0.011 — 지금까지 최고 기록.
# M3(ResNet50)에서 기본 AUG는 평균 성능은 거의 안 바꾸면서(0.633->0.628) fold 간 internal
# 분산을 30% 줄였다(std 0.063->0.045) — "값의 크기"가 아니라 "값의 안정성"에 기여했었다.
# blur를 세게 하지 않은 원래 AUG로도 이 패턴이 재현되는지 확인한다.
#
# --time=24h: UNI+real-time AUG 조합 fold당 실측 시간 — 1차(--strong-blur) 버전이 fold당
# 대략 3~7시간 정도로 완주됐다(fold0 08:24 시작, fold4 11:31 시작해서 24h 안에 전부 끝남) —
# blur 세기만 바꾼 이번 버전도 비슷한 시간대일 것으로 예상.
#
# 완료 후 집계(model_prefix에 _STG_R_AUG_DISP_COX_ADD 태그가 들어간다는 점 주의, _BLUR 없음):
#   python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD
#
# 제출: sbatch sbatch/pma_uni2_coxadd_stg_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

FOLD=$SLURM_ARRAY_TASK_ID
log=".logs/train_tcga_seed84_PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== PMA(uni2,cox_add,STG+R) fold=${FOLD}/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed 84 \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \ --image --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
    --fold "${FOLD}" --n-folds 5 --group-ts 0807pma_uni2_coxadd_stg_kfold5_array_seed84 2>&1 | tee "${log}"
echo "=== PMA(uni2,cox_add,STG+R) fold=${FOLD}/5 Complete: $(date) ==="
