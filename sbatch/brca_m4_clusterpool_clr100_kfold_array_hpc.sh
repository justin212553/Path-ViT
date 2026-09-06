#!/bin/bash
#SBATCH --job-name=PVT-BRCA-cp-clr100
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_clusterpool_clr100_kfold_array_%a.log

# 2026-09-05: sbatch/brca_m4_clusterpool_clr100_fold0seed84_test_hpc.sh 단일 fold 테스트 결과
# — test_c_index=0.7539 (best checkpoint epoch 8), 같은 fold의 M7(0.7367)을 처음으로 실제
# 넘었다(지금까지 PAAD/BRCA 통틀어 WSI 포함 모델이 같은 fold에서 M7을 이긴 적이 없었음).
# 10-fold 전체로 확정 검증.
#
# 레시피 = sbatch/brca_m4_clusterpool_kfold_array_hpc.sh(순정 cluster_pool) + --clinical-lr-mult 100.
# CNV/mutation은 아직 미포함(BRCA 데이터/코드 자체가 없음 — PDAC 전용 하드코딩, 포팅 필요,
# 2026-09-05 확인) — 사용자 결정으로 CLR100부터 먼저 검증.
#
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5. seed=84/126, fold=0..4.
# fold0/seed84는 이미 단일 테스트로 완료됨 — 이 array job도 그대로 재실행한다(재현성 확인 겸,
# 굳이 스킵 로직 안 넣음 — 어차피 checkpoint는 덮어써지고 시간도 크게 안 걸림).
#
# 완료 후(10개 fold 로그 확인, .logs/kfold_preds/에 CSV 10개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_SS_AUX_CLUSTERPOOL_CLR100 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/paired_bootstrap_delta.py --split internal --dataset brca \
#       --model-a BRCA_M7_CONS882_STG --model-b BRCA_PMA_CONS882_STG_SS_AUX_CLUSTERPOOL_CLR100 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/brca_m4_clusterpool_clr100_kfold_array_hpc.sh

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

log=".logs/train_brca_m4_clusterpool_clr100_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA cluster_pool+CLR100 seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --clinical-staging --cluster-pool --clinical-lr-mult 100 \
    --external-tss none --group-ts 0905_brca_m4_clusterpool_clr100_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA cluster_pool+CLR100 seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
