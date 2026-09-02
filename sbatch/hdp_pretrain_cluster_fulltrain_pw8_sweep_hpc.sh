#!/bin/bash
#SBATCH --job-name=PVT-HDPPC-fulltrain
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/hdp_pretrain_cluster_fulltrain_pw8_array_%a.log

# 2026-09-02: M7/PMA에 이어 HDP_Pretrain_Cluster(WSI+RNA+Clinical, PanNuke 지도학습 종양함량 +
# 군집 CNN/MLP)를 같은 "전체 TCGA train + 고정epoch + external만 + 5시드" 프로토콜로 검증한다
# — internal pooled k-fold c-index의 신뢰성 문제(findings_backlog.md 2026-09-02) 때문.
# pathway8(leakage 없는 RNA 패널)만 쓴다 — literature_1500(leak)은 M7/PMA에서 이미 충분히
# 확인됨. clinical-lr-mult는 어젯밤 M7 sweep에서 최고점이었던 10 하나만(사용자 지시, 스윕 안 함).
#
# 2개 설정 x 5시드(42/84/126/168/210) = 10 array task.
# SLURM_ARRAY_TASK_ID(0~9) -> config_idx = id/5, seed_idx = id%5.
#
# [사전 준비] uni2native WSI feature가 HPC에 이미 있어야 함(M1~M7/기존 HDP 계열이 이미 이걸로
# 돌아갔으므로 있을 것으로 예상 — data/patches_{tcga,cptac}/tiles/*/{features_uni2native.pt,
# coords_uni2native.pt}). data/hdp_pretrain_tumor_content_head.pt(PanNuke 학습된 종양함량 head,
# git-tracked)도 필요.
#
# 완료 후(.logs/external_preds/에 cptac_HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8[_CLR10]_
# FULLTRAIN_seed*.csv 10개 확인):
#   python scripts/pool_fulltrain_external_preds.py --dataset cptac \
#       --model HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8 --seeds 42,84,126,168,210 --bootstrap 2000
#   python scripts/pool_fulltrain_external_preds.py --dataset cptac \
#       --model HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8_CLR10 --seeds 42,84,126,168,210 --bootstrap 2000
#   # M7(RNA+Clinical만, pathway8) 대비 WSI+cluster 추가 유의성:
#   python scripts/paired_bootstrap_delta_fulltrain.py --dataset cptac \
#       --model-a M7_PW8_STG_R_COX_ADD --model-b HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8 \
#       --seeds 42,84,126,168,210 --bootstrap 2000
#   python scripts/paired_bootstrap_delta_fulltrain.py --dataset cptac \
#       --model-a M7_PW8_STG_R_COX_ADD_CLR10 --model-b HDP_PRETRAIN_CLUSTER_PW8_STG_R_GROWTH8_CLR10 \
#       --seeds 42,84,126,168,210 --bootstrap 2000
#
# 제출: sbatch sbatch/hdp_pretrain_cluster_fulltrain_pw8_sweep_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(42 84 126 168 210)
N_SEEDS=5
IDX=$SLURM_ARRAY_TASK_ID
CONFIG_IDX=$((IDX / N_SEEDS))
SEED_IDX=$((IDX % N_SEEDS))
SEED=${SEEDS[$SEED_IDX]}

if [ "$CONFIG_IDX" -eq 0 ]; then
  EXTRA_ARGS=""
  TAG="baseline"
else
  EXTRA_ARGS="--clinical-lr-mult 10"
  TAG="CLR10"
fi

log=".logs/train_hdp_pretrain_cluster_fulltrain_pw8_${TAG}_seed${SEED}.log"

echo "=== HDP_Pretrain_Cluster fulltrain pathway8 ${TAG} seed=${SEED} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u train_hdp_pretrain_cluster.py --dataset tcga --external --seed "${SEED}" \
    --rna-genes pathway8 --full-train --epochs 30 ${EXTRA_ARGS} \
    --group-ts hdp_pretrain_cluster_fulltrain_pw8_0902 2>&1 | tee "${log}"
echo "=== HDP_Pretrain_Cluster fulltrain pathway8 ${TAG} seed=${SEED} Complete: $(date) ==="
