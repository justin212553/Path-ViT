#!/bin/bash
#SBATCH --job-name=PVT-porpoise-ownrna-mmf-eval-cptac
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-24
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_ownrna_mmf_eval_cptac_array_%a.log

# 2026-09-06: sbatch/porpoise_ownrna_mmf_train_5seed_array_hpc.sh로 학습한 5seed x 5fold = 25개
# 체크포인트 각각을 CPTAC 전체(external, 학습에 전혀 등장한 적 없는 코호트)에 평가한다
# (porpoise/eval_external.py — PORPOISE 공식 코드엔 없던 기능, 새로 작성).
#
# 선행 조건: porpoise_ownrna_mmf_train_5seed_array_hpc.sh의 25개 seed(5개) 잡이 전부 완료돼
# s_{fold}_checkpoint.pt가 5개씩(총 25개) 있어야 함:
#   ls porpoise/results_ownrna_mmf/5foldcv/*/tcga_paad_s*/s_*_checkpoint.pt | wc -l   # 25이어야 함
#
# 출력 파일명은 scripts/pool_multiseed_external_preds.py가 그대로 읽을 수 있게 그 스크립트의
# 글롭 관례(.logs/external_preds/{dataset}_{model}*_seed{seed}_fold{fold}of{n_folds}.csv)를
# 맞췄다 — 이 project의 다른 external 평가와 동일한 pooling 스크립트를 그대로 재사용 가능:
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PORPOISE_OWNRNA_MMF --seeds 84,126,168,210,252 --n-folds 5 --bootstrap 2000
#
# 제출(train array 완료 확인 후): sbatch sbatch/porpoise_ownrna_mmf_eval_cptac_5seed_array_hpc.sh

cd /pub/wonseukl/Path-ViT/porpoise

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

SEEDS=(84 126 168 210 252)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

OUT_CSV="/pub/wonseukl/Path-ViT/.logs/external_preds/cptac_PORPOISE_OWNRNA_MMF_seed${SEED}_fold${FOLD}of${N_FOLDS}.csv"

echo "=== PORPOISE MMF(own-RNA) CPTAC external eval seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u eval_external.py \
    --seed "${SEED}" --fold "${FOLD}" \
    --results-dir results_ownrna_mmf \
    --tcga-data-root /pub/wonseukl/Path-ViT/porpoise/data_root_true_resnet50 \
    --cptac-data-root /pub/wonseukl/Path-ViT/data/porpoise_style_features/cptac \
    --out-csv "${OUT_CSV}"
echo "=== PORPOISE MMF(own-RNA) CPTAC external eval seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
