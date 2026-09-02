#!/bin/bash
#SBATCH --job-name=PVT-HDPCLU-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/hdp_cluster_multiseed_kfold_array_%a.log

# HDP_CLUSTER(models/hdp_cluster.py, train_hdp_cluster.py) — HDP(비지도 k-means 군집, 결정론적
# 4*K차원 통계, 학습 파라미터 0개)가 M7과 계속 동률이라, 원래 계획에서 "리스크가 크다"고
# 미뤄뒀던 GrowthPatternCNN(침윤전선)과 MaturityMLP(성숙도)를 다시 넣어 152개 생존 라벨로
# end-to-end 학습시켜본다(2026-09-01 사용자 결정).
#
# train_light.py(--HDP/--HDP-PRETRAIN)와 달리 patch/슬라이드 forward가 실제로 있다(coords
# 기반 occupancy map -> CNN, patch feature -> MLP) — 전용 스크립트(train_hdp_cluster.py) 사용,
# train.py급 CNN/ViT 인코더는 없음(uni2native feature가 이미 h5로 추출돼 있어 그걸 그대로 씀).
# WSILookup이 환자당 soft cluster weight를 in-memory 캐싱해 epoch 반복 비용을 줄인다(로컬
# 스모크 테스트로 확인).
#
# ⚠️ 이 job은 --HDP/--HDP-PRETRAIN과 달리 작은 CSV만으로 안 된다 — WSILookup이 data/
# uni2h_official_features/{tcga,cptac}/*.h5(45GB, raw feature+coords)를 직접 읽는다. M1~M7이
# uni2native로 HPC에서 이미 학습됐다면 이 raw h5든 변환된 per-slide .pt 트리든 뭔가는 있겠지만,
# 정확히 이 경로(data/uni2h_official_features/)에 h5 형태로 있는지는 로컬에서 확인 못 했다 —
# 실행 전에 해당 경로 존재 여부를 먼저 확인할 것(없으면 scripts/download_uni2h_official_features.py
# 또는 별도 동기화 필요).
#
# array 관례: SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5 (2seed x 5fold, seed42 제외).
#
# 완료 후(.logs/kfold_preds/에 CSV 10개 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model HDP_CLUSTER_INT1500_STG_R_GROWTH8 --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 그다음 external 평가는 sbatch/hdp_cluster_multiseed_external_eval_hpc.sh(재학습 없이
# eval-external-ckpt로 checkpoint 재사용, 이 10개 학습이 전부 끝난 뒤 제출).
#
# 제출: sbatch sbatch/hdp_cluster_multiseed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_HDP_CLUSTER_INT1500_STG_R_GROWTH8_kfold5_fold${FOLD}.log"

echo "=== HDP_CLUSTER seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u train_hdp_cluster.py --dataset tcga --external --seed "${SEED}" \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --epochs 100 --patience 20 \
    --group-ts 0901hdp_cluster_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== HDP_CLUSTER seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
