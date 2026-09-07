#!/bin/bash
#SBATCH --job-name=PVT-BRCA-porpoise-clr100
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_porpoise_consistency_clr100_kfold_array_%a.log

# 2026-09-06: sbatch/brca_m4_clusterpool_clr100_kfold_array_hpc.sh(cluster_pool+CLR100, M7을
# 처음 이긴 조합 — 단일 fold test_c_index=0.7539 vs M7 0.7367)와 완전히 동일한 레시피(consistency
# RNA/staging/CLR100)에서 --cluster-pool만 빼고 PORPOISE 아키텍처(ABMIL+Kronecker,
# scripts/train_brca_porpoise.py 2026-09-06 확장 — 지금 PAAD 최종 레시피의 WSI 풀링/결합 방식)로
# 교체 — 사용자 지시("Cluster pool 말고, 지금 PADC에서 잘 나오는 PORPOISE식 ABMIL이 들어간
# 걸로 바꾸지").
#
# CNV/mutation은 BRCA 쪽 데이터/코드가 아직 없어 포함 안 함(PDAC 전용). --surv-loss(nll_surv/
# both)도 이 스크립트엔 아직 이식 안 함 — 이번엔 아키텍처(cluster_pool vs PORPOISE ABMIL) 하나만
# 분리해서 비교하는 게 목적.
#
# --external-tss none(2026-08-31 결정과 동일) — 1058명 전체를 internal k-fold 풀로 씀.
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5. seed=84/126, fold=0..4.
# --requeue: free-gpu partition preemption 대비.
#
# 완료 후(10개 fold 로그 확인, .logs/kfold_preds/에 CSV 10개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PORPOISE_CONS882_SS_DISP_STG_CLR100 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/paired_bootstrap_delta.py --split internal --dataset brca \
#       --model-a BRCA_PMA_CONS882_STG_SS_AUX_CLUSTERPOOL_CLR100 \
#       --model-b BRCA_PORPOISE_CONS882_SS_DISP_STG_CLR100 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 train_brca_porpoise.py의 model_prefix 접미사 순서를 직접 뽑은 것 —
# `ls .logs/kfold_preds/brca_BRCA_PORPOISE*` 로 확인 권장)
#
# 제출: sbatch sbatch/brca_porpoise_consistency_clr100_kfold_array_hpc.sh

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

log=".logs/train_brca_porpoise_consistency_clr100_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA PORPOISE(consistency,STG,CLR100) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_porpoise --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --clinical-staging --clinical-lr-mult 100 \
    --external-tss none --group-ts 0906_brca_porpoise_consistency_clr100_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA PORPOISE(consistency,STG,CLR100) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
