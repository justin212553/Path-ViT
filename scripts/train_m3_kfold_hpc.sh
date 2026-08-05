#!/bin/bash
#SBATCH --job-name=PVT-M3-kfold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m3_kfold.log

# train_m1_kfold_hpc.sh와 동일한 이유(GPU 5개 동시 점유가 어려워 단일 GPU 순차 실행으로 전환).
# --time=24:00:00: free-gpu가 24시간 넘는 요청은 큐에 아예 안 올려준다 — 5-fold 실측 추정
# ~20시간 대비 여유가 4시간뿐이라 중간에 잘릴 수 있다(잘리면 --fold N만 다시 제출).
# M3(WSI+RNA, clinical 제외, ViT_PMA --no-clinical) — RNA가 있으니 EX+SS+AUX+AUG+DISP 전부.
#
# 완료 후 집계(model_prefix에 --no-clinical의 _NOCLINICAL 태그가 들어간다는 점 주의):
#   python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model PMA_EX_SS_AUX_AUG_NOCLINICAL_DISP
#
# 제출: sbatch scripts/train_m3_kfold_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

Folds=(0 1 2 3 4)

for fold in "${Folds[@]}"; do
    log=".logs/train_tcga_seed84_PMA_EX_SS_AUX_AUG_NOCLINICAL_DISP_kfold5_fold${fold}.log"
    echo "=== M3(PMA_EX_SS_AUX_AUG_NOCLINICAL_DISP) fold=${fold}/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
    python -u ./train.py --PMA --no-clinical --rna-genes literature_1500 --dataset tcga --external --seed 84 \
        --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
        --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
        --fold "${fold}" --n-folds 5 --group-ts 0804m3_kfold5_seq 2>&1 | tee "${log}"
    echo "=== M3(PMA_EX_SS_AUX_AUG_NOCLINICAL_DISP) fold=${fold}/5 Complete: $(date) ==="
done

echo "=== ALL M3 K-FOLD RUNS COMPLETE: $(date) ==="
