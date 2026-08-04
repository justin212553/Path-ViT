#!/bin/bash
#SBATCH --job-name=PVT-M1-kfold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m1_kfold_array_%a.log

# train_m1_hpc.sh(단일 6:2:2 split)의 K-fold 버전 — scripts/_pma_ex_ss_aux_tileaugment_dispersion_
# kfold_array.sh와 동일한 job array 패턴(--array=0-4, $SLURM_ARRAY_TASK_ID를 --fold로).
# 2026-08-04 실측: PMA fold 1개(30 epoch) 완주에 ~4시간 — free-gpu에 A30 5개가 다 비어있으면
# 5-fold가 사실상 동시에 돌아 벽시계 기준 fold 1개 시간(~4시간)으로 끝난다. --time=6h는 그
# 실측치에 여유를 더한 값(M1은 RNA/co-attention이 없어 PMA보다 가볍고, 4h보다 오래 걸리진
# 않을 것으로 예상).
#
# SS(patch dropout)+AUG(실시간 augmentation)+DISP(attention dispersion) — EX/AUX는 RNA
# 브랜치가 없는 M1엔 대응 항목 없어 제외(train_m1_hpc.sh와 동일한 이유).
#
# 완료 후 5개 fold 예측 풀링(로그인 노드, 인터넷 불필요):
#   python scripts/pool_kfold_preds.py --dataset tcga --model M1_SS_AUG_DISP --seed 42 --n-folds 5
#
# 제출: sbatch scripts/train_m1_kfold_hpc.sh
# 진행 확인: squeue -u $USER  (JobID_0 ~ JobID_4로 5줄)

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
