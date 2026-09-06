#!/bin/bash
#SBATCH --job-name=PVT-BRCA-clusterpool
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_clusterpool_kfold_array_%a.log

# 2026-09-05: PAAD에서 Nystrom(oversmoothing 무죄로 확인)/ABMIL(gradient가 weight_decay 유무와
# 무관하게 전혀 안 닿는 dead module로 확인)을 둘 다 우회하는 cluster_pool(models/vit_pma.py) —
# raw feature 공간 unsupervised 군집(K개) 대표 토큰을 RNA co-attention에 바로 넘기는 구조 —
# 이 M7과 통계적으로 동등한 수준까지 WSI의 "해로움"을 없앴다(external delta -0.0089, p=0.684,
# 10-fold paired bootstrap). 다만 아직 M7 대비 유의한 *양의* 기여는 못 보였다.
#
# PAAD는 N~110~150명이라 통계 검정력이 낮다 — BRCA(N=1058, ~10배)에서 같은 구조를 재검증하면
# "PAAD에서 안 잡히는 게 표본 크기 문제인지, 진짜 WSI에 신호가 없는 건지"를 훨씬 강한 검정력
#으로 가릴 수 있다(사용자 결정, 2026-09-05).
#
# 선행 조건: sbatch/fit_clusters_brca_uni_hpc.sh를 먼저 돌려서 data/cluster_centroids_brca_uni.pt
# 가 있어야 한다 — 없으면 즉시 FileNotFoundError.
#
# 레시피는 1-layer/2-layer BRCA 실험과 동일 기반(scripts/train_brca_m4.py, --gene-selection
# consistency --clinical-staging --external-tss none)에 --cluster-pool만 추가 — Nystrom도
# ABMIL도 안 쓰므로 --num-transformer-layers는 무의미(지정 안 함).
#
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5. seed=84/126, fold=0..4.
#
# 완료 후(10개 fold 로그 확인, .logs/kfold_preds/에 CSV 10개 있는지 확인):
#   주의(2026-09-05 정정) — --patch-keep-frac 기본값이 0.8(<1.0)이고 --rna-aux-weight
#   기본값이 1.0(>0)이라 명시적으로 안 꺼도 model_prefix에 _SS_AUX가 자동으로 붙는다
#   (BRCA_PMA_CONS882_STG_CLUSTERPOOL이 아니라 BRCA_PMA_CONS882_STG_SS_AUX_CLUSTERPOOL).
#   그리고 --external-tss none이라 external 평가 자체가 없다(1-layer/2-layer 기존 BRCA
#   실험과 동일 관례) — pool_multiseed_external_preds.py는 해당 없음, internal만.
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_SS_AUX_CLUSTERPOOL --seeds 84,126 --n-folds 5 --bootstrap 2000
#   (M7은 이미 완료된 .logs/kfold_preds/brca_BRCA_M7_CONS882_STG_seed{84,126}_fold{0-4}of5.csv 재사용)
#   python scripts/paired_bootstrap_delta.py --split internal --dataset brca \
#       --model-a BRCA_M7_CONS882_STG --model-b BRCA_PMA_CONS882_STG_SS_AUX_CLUSTERPOOL \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출(fit_clusters 완료 확인 후): sbatch sbatch/brca_m4_clusterpool_kfold_array_hpc.sh

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

log=".logs/train_brca_m4_clusterpool_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA cluster_pool seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --clinical-staging --cluster-pool \
    --external-tss none --group-ts 0905_brca_m4_clusterpool_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA cluster_pool seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
