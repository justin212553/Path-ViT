#!/bin/bash
#SBATCH --job-name=PVT-PMA-kfold-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_kfold_array_%a.log

# _pma_ex_ss_aux_tileaugment_dispersion_kfold_ext.sh(단일 GPU 순차 5-fold, ~20시간)의 병렬
# 버전 — train_m1_kfold_array_hpc.sh와 동일한 이유/패턴(야간처럼 free-gpu가 한산한 시간대에
# A30 5개가 동시에 비어있을 걸 노려서 벽시계 기준 fold 1개 시간(~4~6시간)으로 끝내려는 용도).
# $SLURM_ARRAY_TASK_ID(0~4)를 --fold로 넘겨 5개의 독립된 job으로 제출된다 — 5개가 다 못 뜨면
# SLURM이 자리 나는 대로 채워서 돌리므로(순차보다 늦어지지 않음), 밤에 던져두고 아침에
# 확인하는 용도로도 안전하다.
#
# 2026-08-04: 여러 fold가 train/val pool을 상당 부분 공유해(fold마다 test 20%만 바뀜) 동시에
# 같은 타일을 디스크 캐시(data/tile_cache_512/)에 쓰려는 race condition을 이 스크립트의 첫
# 시도에서 실제로 겪었다(UnidentifiedImageError, 5개 중 3개 크래시) — data/patch_utils.py에
# 원자적 쓰기(임시파일+os.replace) + 읽기 실패 시 자동 복구(깨진 캐시를 원본부터 다시 디코딩)를
# 넣어 고쳤다. 지금은 병렬로 돌려도 안전하다.
#
# PMA(clinical 포함, M4 슬롯) — EX+SS+AUX+AUG+DISP 전부.
#
# 완료 후 집계(5개 다 끝난 걸 확인하고, 로그인 노드에서 인터넷 불필요):
#   python scripts/summarize_kfold.py --dataset tcga --seed 42 --n-folds 5 --model PMA_EX_SS_AUX_AUG_DISP
#
# 제출: sbatch scripts/train_pma_kfold_array_hpc.sh
# 진행 확인: squeue -u $USER  (JobID_0 ~ JobID_4로 5줄, ST 컬럼이 R인지 PD인지로 실제 동시성 확인)

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

FOLD=$SLURM_ARRAY_TASK_ID
log=".logs/train_tcga_seed42_PMA_EX_SS_AUX_AUG_DISP_kfold5_fold${FOLD}.log"

echo "=== PMA_EX_SS_AUX_AUG_DISP fold=${FOLD}/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --dataset tcga --seed 42 --PMA --rna-genes literature_1500 \
    --patch-keep-frac 0.8 --rna-aux-weight 1.0 --image --tile-augment --attn-dispersion \
    --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
    --fold "${FOLD}" --n-folds 5 \
    --external --group-ts 0804pma_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA_EX_SS_AUX_AUG_DISP fold=${FOLD}/5 Complete: $(date) ==="
