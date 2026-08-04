#!/bin/bash
#SBATCH --job-name=PVT-M1-kfold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m1_kfold.log

# 2026-08-04: 원래 SLURM job array(--array=0-4, GPU 5개 동시 요청)로 만들었으나, free-gpu
# 파티션에서 A30 5개를 동시에 점유하기가 현실적으로 어렵다는 판단으로 단일 GPU에서 fold
# 0->1->2->3->4를 순차로 도는 버전으로 바꿨다(scripts/_pma_ex_ss_aux_tileaugment_dispersion_
# kfold_array.sh에서 실제로 겪은 디스크 캐시 race condition — 여러 fold가 동시에 같은 타일을
# data/tile_cache_512/에 쓰다 UnidentifiedImageError로 3/5 크래시 — 도 순차 실행이면 애초에
# 안 생긴다. 다만 그 race 자체는 data/patch_utils.py에 원자적 쓰기+자동 복구로 이미 고쳐뒀으니
# 나중에 다시 병렬로 시도해도 안전하다).
#
# fold 1개(30 epoch) 실측 ~4시간 x 5 = ~20시간 예상 — free-gpu 파티션이 24시간 넘는 --time은
# 아예 큐에 안 올려줘서(제출 자체가 거부됨) 24:00:00으로 맞춘다. 20시간 추정치에 여유가 4시간
# 밖에 안 남아 fold 하나가 유독 오래 걸리면 중간에 잘릴 위험이 있다 — 잘리면 그 fold부터
# --fold N만 다시 제출하면 된다(이전 fold 결과는 체크포인트/CSV로 이미 남아있어 안 날아감).
#
# SS(patch dropout)+AUG(실시간 augmentation)+DISP(attention dispersion) — EX/AUX는 RNA
# 브랜치가 없는 M1엔 대응 항목 없어 제외(train_m1_hpc.sh와 동일한 이유).
#
# 완료 후 5개 fold 예측 집계(internal pooled + external 평균, 로그인 노드에서 인터넷 불필요):
#   python scripts/summarize_kfold.py --dataset tcga --seed 42 --n-folds 5 --model M1_SS_AUG_DISP
#
# 제출: sbatch scripts/train_m1_kfold_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

Folds=(0 1 2 3 4)

for fold in "${Folds[@]}"; do
    log=".logs/train_tcga_seed42_M1_SS_AUG_DISP_kfold5_fold${fold}.log"
    echo "=== M1_SS_AUG_DISP fold=${fold}/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
    python -u ./train.py --M1 --dataset tcga --external --seed 42 \
        --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion \
        --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
        --fold "${fold}" --n-folds 5 --group-ts 0804m1_kfold5_seq 2>&1 | tee "${log}"
    echo "=== M1_SS_AUG_DISP fold=${fold}/5 Complete: $(date) ==="
done

echo "=== ALL M1_SS_AUG_DISP K-FOLD RUNS COMPLETE: $(date) ==="
