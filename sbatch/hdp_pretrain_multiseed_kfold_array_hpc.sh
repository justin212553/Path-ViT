#!/bin/bash
#SBATCH --job-name=PVT-HDPPRE-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/hdp_pretrain_multiseed_kfold_array_%a.log

# HDP_PRETRAIN(models/hdp.py, train_light.py --HDP-PRETRAIN) — --HDP(비지도 k-means 군집)이
# 여러 정제(hard/soft weighting, dispersion/heterogeneity/atypicality 추가)를 거쳐도 M7과
# 통계적으로 계속 동률이어서(2026-09-01), 사용자 결정으로 진짜 라벨 기반 supervision으로
# 전환했다 — PanNuke(pancreas subset 195장, 핵 단위 라벨)로 학습시킨 종양 함량 회귀 head
# (scripts/train_hdp_pretrain_head.py, held-out val_corr=0.60 — 이번 세션에서 시도한 다른
# 어떤 방법보다 강한 신호)를 우리 코호트에 적용해(scripts/apply_hdp_pretrain_head.py) 얻은
# 4차원 통계(평균/heterogeneity/dispersion/고함량비율)를 M7에 cox_add로 추가한다.
#
# --HDP와 마찬가지로 patch forward가 전혀 없다(feature는 이미 사전 계산돼 CSV로 커밋됨:
# data/tumor_content_uni2native_{tcga,cptac}.csv) — train.py가 아니라 train_light.py 사용.
#
# ⚠️ 이 job은 git pull만으로 충분하다 — data/tumor_content_uni2native_{tcga,cptac}.csv가
# 이미 git에 커밋돼 있어서(PanNuke 원본이나 학습된 head checkpoint를 HPC에 따로 옮길 필요
# 없음, 그 무거운 단계는 로컬에서 이미 끝내고 최종 4차원 CSV만 커밋했다).
#
# array 관례: SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5 (2seed x 5fold, seed42 제외).
#
# 완료 후(.logs/kfold_preds/에 CSV 10개 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model HDP_PRETRAIN_INT1500_STG_R --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 그다음 external 평가는 sbatch/hdp_pretrain_multiseed_external_eval_hpc.sh(재학습 없이
# eval-external-ckpt로 checkpoint 재사용, 이 10개 학습이 전부 끝난 뒤 제출).
#
# 제출: sbatch sbatch/hdp_pretrain_multiseed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_HDP_PRETRAIN_INT1500_STG_R_kfold5_fold${FOLD}.log"

echo "=== HDP_PRETRAIN seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u train_light.py --HDP-PRETRAIN --dataset tcga --external --seed "${SEED}" \
    --rna-genes literature_1500_intersection \
    --clinical-margin --clinical-staging \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --epochs 100 --patience 20 \
    --group-ts 0901hdp_pretrain_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== HDP_PRETRAIN seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
