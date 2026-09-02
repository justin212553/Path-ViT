#!/bin/bash
#SBATCH --job-name=PVT-HDPPC-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/hdp_pretrain_cluster_multiseed_kfold_array_%a.log

# HDP_PRETRAIN_CLUSTER(train_hdp_pretrain_cluster.py) — HDP_Pretrain(PanNuke로 학습된 진짜
# 종양 함량 head, 4개 결정론적 요약 통계, held-out val_corr=0.60이지만 M7과 여전히 동률:
# internal -0.0128/external +0.0040)이 "정보를 너무 압축한 거 아니냐"는 질문(사용자)에서 시작.
# HDP_Cluster(K=10 비지도 군집 버전)에 CNN+MLP를 넣었을 때도 무변화였던 전례가 있어(순수 군집
# 버전 대비 internal +0.0013/external +0.0005), 여기서는 "진짜 라벨 기반 신호"에 대해서도
# 압축 여부가 무관한지 직접 검증한다 — models/hdp_cluster.py::HDPCluster를 K=1(군집 10개
# 대신 종양 함량 스칼라 하나)로 재사용, GrowthPatternCNN이 4개 요약 통계로 뭉개지 않은 전체
# 공간 map을 직접 본다.
#
# ⚠️ train_hdp_cluster.py와 동일하게 WSISurvivalDataset(feature_backbone="uni2native")을 쓴다
# — M1~M7이 이미 이 backbone으로 HPC에서 학습된 적 있으니 정상 동작할 것으로 기대하지만, 이
# 스크립트 자체는 로컬에서 데이터 부재로 끝까지 검증 못 했다(모델 구성까지는 확인 완료).
#
# array 관례: SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5 (2seed x 5fold, seed42 제외).
#
# 완료 후(.logs/kfold_preds/에 CSV 10개 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model HDP_PRETRAIN_CLUSTER_INT1500_STG_R_GROWTH8 --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 그다음 external 평가는 sbatch/hdp_pretrain_cluster_multiseed_external_eval_hpc.sh(재학습 없이
# eval-external-ckpt로 checkpoint 재사용, 이 10개 학습이 전부 끝난 뒤 제출).
#
# 제출: sbatch sbatch/hdp_pretrain_cluster_multiseed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_HDP_PRETRAIN_CLUSTER_INT1500_STG_R_GROWTH8_kfold5_fold${FOLD}.log"

echo "=== HDP_PRETRAIN_CLUSTER seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u train_hdp_pretrain_cluster.py --dataset tcga --external --seed "${SEED}" \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --epochs 100 --patience 20 \
    --group-ts 0901hdp_pretrain_cluster_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== HDP_PRETRAIN_CLUSTER seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
