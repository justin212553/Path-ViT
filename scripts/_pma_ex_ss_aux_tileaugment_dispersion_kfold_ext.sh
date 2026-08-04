#!/bin/bash
#SBATCH --job-name=PVT-PMA-kfold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_kfold.log

# 2026-08-04: scripts/_pma_ex_ss_aux_tileaugment_dispersion_kfold_array.sh(SLURM job array,
# --array=0-4, GPU 5개 동시 요청)를 대체 — free-gpu 파티션에서 A30 5개를 동시에 점유하기가
# 현실적으로 어렵다는 판단으로, 단일 GPU에서 fold 0->1->2->3->4를 순차로 도는 버전으로 바꿨다.
# (그 array 버전 실행 중 실제로 겪은 디스크 캐시 race condition — 여러 fold가 동시에 같은
# 타일을 data/tile_cache_512/에 쓰다 UnidentifiedImageError로 5개 중 3개 크래시 — 도 순차
# 실행이면 애초에 안 생긴다. race 자체는 data/patch_utils.py에 원자적 쓰기+자동 복구로 이미
# 고쳐뒀으니 나중에 다시 병렬로 시도해도 안전하다.)
#
# fold 1개(30 epoch) 실측 ~4시간 x 5 = ~20시간 예상 — free-gpu 파티션이 24시간 넘는 --time은
# 아예 큐에 안 올려줘서(제출 자체가 거부됨) 24:00:00으로 맞춘다. 20시간 추정치에 여유가 4시간
# 밖에 안 남아 fold 하나가 유독 오래 걸리면 중간에 잘릴 위험이 있다 — 잘리면 그 fold부터
# --fold N만 다시 제출하면 된다(이전 fold 결과는 체크포인트/CSV로 이미 남아있어 안 날아감).
#
# PMA(clinical 포함, M4 슬롯) — EX+SS+AUX+AUG+DISP 전부.
#
# 완료 후 집계:
#   python scripts/summarize_kfold.py --dataset tcga --seed 42 --n-folds 5 --model PMA_EX_SS_AUX_AUG_DISP
#
# 제출: sbatch scripts/_pma_ex_ss_aux_tileaugment_dispersion_kfold_ext.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

Folds=(0 1 2 3 4)

for fold in "${Folds[@]}"; do
    log=".logs/train_tcga_seed42_PMA_EX_SS_AUX_AUG_DISP_kfold5_fold${fold}.log"
    echo "=== PMA_EX_SS_AUX_AUG_DISP fold=${fold}/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
    python -u ./train.py --dataset tcga --seed 42 --PMA --rna-genes literature_1500 \
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 --image --tile-augment --attn-dispersion \
        --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
        --fold "${fold}" --n-folds 5 --group-ts 0804pma_kfold5_seq 2>&1 | tee "${log}"
    echo "=== PMA_EX_SS_AUX_AUG_DISP fold=${fold}/5 Complete: $(date) ==="
done

echo "=== ALL PMA_EX_SS_AUX_AUG_DISP K-FOLD RUNS COMPLETE: $(date) ==="
