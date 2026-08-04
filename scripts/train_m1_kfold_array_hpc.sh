#!/bin/bash
#SBATCH --job-name=PVT-M1-kfold-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m1_kfold_array_%a.log

# train_m1_kfold_hpc.sh(단일 GPU 순차 5-fold, ~20시간)의 병렬 버전 — free-gpu가 한산한
# 시간대(야간 등)에 A30 5개가 동시에 비어있을 걸 노려서 벽시계 기준 fold 1개 시간(~4~6시간)
# 으로 끝내려는 용도. $SLURM_ARRAY_TASK_ID(0~4)를 --fold로 넘겨 5개의 독립된 job으로 제출된다
# — 5개가 다 못 뜨면 SLURM이 자리 나는 대로 채워서 돌리므로(순차보다 늦어지지 않음), 밤에
# 던져두고 아침에 확인하는 용도로도 안전하다.
#
# 2026-08-04: 여러 fold가 train/val pool을 상당 부분 공유해(fold마다 test 20%만 바뀜) 동시에
# 같은 타일을 디스크 캐시(data/tile_cache_512/)에 쓰려는 race condition을 실제로 겪었다
# (UnidentifiedImageError, 5개 중 3개 크래시) — data/patch_utils.py에 원자적 쓰기(임시파일+
# os.replace) + 읽기 실패 시 자동 복구(깨진 캐시를 원본부터 다시 디코딩)를 넣어 고쳤다. 지금은
# 병렬로 돌려도 안전하다.
#
# --time=6h: array 태스크 1개당 fold 1개(~4시간 실측)만 하면 되므로 순차 버전(24h)보다 훨씬
# 짧게 잡아도 된다 — 짧은 --time이 스케줄러 backfill에서 우선순위/여유 슬롯을 더 잘 찾는 경향이
# 있어 유휴 시간대 노리는 전략과도 맞는다.
#
# SS+AUG+DISP — EX/AUX는 RNA 브랜치가 없는 M1엔 대응 항목 없어 제외.
#
# 완료 후 집계(5개 다 끝난 걸 확인하고, 로그인 노드에서 인터넷 불필요):
#   python scripts/summarize_kfold.py --dataset tcga --seed 42 --n-folds 5 --model M1_SS_AUG_DISP
#
# 제출: sbatch scripts/train_m1_kfold_array_hpc.sh
# 진행 확인: squeue -u $USER  (JobID_0 ~ JobID_4로 5줄, ST 컬럼이 R인지 PD인지로 실제 동시성 확인)

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

FOLD=$SLURM_ARRAY_TASK_ID
log=".logs/train_tcga_seed42_M1_SS_AUG_DISP_kfold5_fold${FOLD}.log"

echo "=== M1_SS_AUG_DISP fold=${FOLD}/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M1 --dataset tcga --external --seed 42 \
    --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion \
    --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
    --fold "${FOLD}" --n-folds 5 --group-ts 0804m1_kfold5_array 2>&1 | tee "${log}"
echo "=== M1_SS_AUG_DISP fold=${FOLD}/5 Complete: $(date) ==="
