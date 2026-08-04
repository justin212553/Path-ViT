#!/bin/bash
#SBATCH --job-name=PVT-PMA-kfold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_kfold_array_%a.log

# 2026-08-04: 5-fold를 한 job 안에서 순차로(fold 0→1→2→3→4) 돌리면 fold 1개 시간의 5배가
# 그대로 걸린다(로컬 실측 기준 fold당 ~6.5~7.5시간 → 순차 5-fold ~33~38시간). 대신 SLURM job
# array(--array=0-4)로 fold마다 독립된 job을 5개 띄우면, free-gpu 파티션에 A30가 5개 다
# 비어있을 때 사실상 동시에 돌아 벽시계 기준 fold 1개 시간(~6~7시간)으로 끝날 수 있다 — 물론
# 실제 동시성은 그 순간 파티션에 A30가 몇 개 비어있느냐에 달려있고, 부족하면 SLURM이 나머지를
# 큐에 대기시켰다가 자리 나는 대로 돌린다(그래도 순차 강제보다는 항상 같거나 빠르다).
#
# $SLURM_ARRAY_TASK_ID가 0~4로 각 태스크에 자동 배정되고, 이걸 --fold로 그대로 넘긴다 —
# data/dataset.py::_kfold_case_split이 같은 seed로 fold 분할을 재현하므로, 5개 태스크가 서로
# 완전히 겹치지 않는 test 20%씩을 보게 된다(대화에서 이미 검증된 로직, 새로 만든 게 아님).
#
# 여러 fold가 train/val pool을 상당 부분 공유하므로(fold마다 test 20%만 바뀌고 나머지 80%는
# 대부분 겹침), data/patch_utils.py::build_tile_cache의 디스크 JPEG 캐시(TILE_DISK_CACHE_DIR)를
# 여러 태스크가 동시에 같은 파일에 쓰려고 하는 드문 경합이 있을 수 있다 — 이론상 아주 좁은
# 타이밍 창에서 한쪽이 쓰는 도중 다른 쪽이 읽다 PIL 디코딩 에러를 낼 가능성이 있는데, 실제로
# 한 태스크가 죽으면 그 fold만 재제출하면 된다(다른 fold는 무관하게 영향 없음).
#
# 완료 후 5개 fold 예측을 모으려면(로그인 노드, 인터넷 필요 없음):
#   python scripts/pool_kfold_preds.py --dataset tcga --model PMA_EX_SS_AUX_AUG_DISP --seed 42 --n-folds 5
#
# 제출: sbatch scripts/_pma_ex_ss_aux_tileaugment_dispersion_kfold_array.sh
# 진행 확인: squeue -u $USER  (JobID가 <jobid>_0 ~ <jobid>_4로 5줄 뜬다)

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

FOLD=$SLURM_ARRAY_TASK_ID
log=".logs/train_tcga_seed42_PMA_EX_SS_AUX_AUG_DISP_kfold5_fold${FOLD}.log"

echo "=== PMA_EX_SS_AUX_AUG_DISP fold=${FOLD}/5 Start: $(date) (job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}, node $(hostname)) ==="
python -u ./train.py --dataset tcga --seed 42 --PMA --rna-genes literature_1500 \
    --patch-keep-frac 0.8 --rna-aux-weight 1.0 --image --tile-augment --attn-dispersion \
    --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
    --fold "${FOLD}" --n-folds 5 \
    --external --group-ts 0804pma_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA_EX_SS_AUX_AUG_DISP fold=${FOLD}/5 Complete: $(date) ==="
