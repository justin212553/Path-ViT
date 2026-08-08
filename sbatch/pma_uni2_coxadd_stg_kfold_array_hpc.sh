#!/bin/bash
#SBATCH --job-name=PVT-PMA-uni2-coxadd-stg-kfold-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_uni2_coxadd_stg_kfold_array_%a.log

# 2026-08-08: 원래 이 스크립트는 --tile-augment 없이 --image만 켜져 있었다 — 즉 real-time
# augmentation은 애초에 안 쓰고 있었는데도 매 epoch마다 raw jpg를 다시 디코딩해 frozen UNI2-h
# backbone을 다시 forward하고 있었다(--tile-decode-workers/--cache-val-tiles/--cache-external-
# tiles까지 딸려 붙어 128G RAM을 씀). frozen backbone 출력은 epoch마다 동일하므로 결과는
# sbatch/extract_features_uni2_array_hpc.sh로 이미 뽑아둔 features_uni2.pt(precomputed 모드,
# --image 없음)를 쓰는 것과 완전히 같고, --image만 훨씬 느리다 — 그래서 --image/--tile-decode-
# workers/--cache-val-tiles/--cache-external-tiles를 전부 제거하고 precomputed 모드로 바로
# 읽게 고쳤다(model_prefix/체크포인트/집계 명령은 --tile-augment를 쓴 적이 없어 그대로 유지된다).
#
# 로컬 no-aug 파일럿(PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD, seed84) 기준값:
#   internal(pooled) 0.617 / external(mean) 0.644 ± 0.011 — 지금까지 최고 기록.
#
# --time=24h: 기존 --image 경로(raw jpg 재디코딩+backbone 재forward) 기준 fold당 3~7시간
# 실측치라 안전 마진용으로 뒀던 값 — precomputed 모드는 CNN/ViT backbone forward 자체가 없어
# 훨씬 빨라질 것으로 예상되지만 정확한 실측 전까지는 그대로 24h 유지(안전 마진).
#
# 완료 후 집계(model_prefix에 _STG_R_DISP_COX_ADD 태그, _AUG 없음):
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
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --fold "${FOLD}" --n-folds 5 --group-ts 0807pma_uni2_coxadd_stg_kfold5_array_seed84 2>&1 | tee "${log}"
echo "=== PMA(uni2,cox_add,STG+R) fold=${FOLD}/5 Complete: $(date) ==="
