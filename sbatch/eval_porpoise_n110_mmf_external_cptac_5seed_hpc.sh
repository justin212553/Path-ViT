#!/bin/bash
#SBATCH --job-name=PVT-porpoise-n110-eval
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-24
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/eval_porpoise_n110_mmf_external_cptac_5seed_array_%a.log

# 2026-10-10: run_porpoise_n110_mmf_5seed_array_hpc.sh(results_n110_mmf)의 CPTAC-PDAC external 평가.
# 각 (seed, fold) 체크포인트를 CPTAC 136명 전체에 적용. eval_external.py가 그 fold의 TCGA train split
# (splits/5foldcv_n110/tcga_paad_seed{seed}/splits_{fold}.csv)으로 StandardScaler를 재구성해 CPTAC raw
# RNA에 적용한다 — 우리 --fold-safe의 "CPTAC은 TCGA train 통계로 정규화"와 같은 규칙.
# array index = seed_idx*5 + fold. 출력은 우리 예측 CSV 형식 그대로 .logs/external_preds/.

cd /pub/wonseukl/Path-ViT/porpoise
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

SEEDS=(84 126 42 168 210)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((IDX / N_FOLDS))]}
FOLD=$((IDX % N_FOLDS))
OUT_CSV="/pub/wonseukl/Path-ViT/.logs/external_preds/cptac_PORPOISE_N110_MMF_seed${SEED}_fold${FOLD}of${N_FOLDS}.csv"

echo "=== PORPOISE N110 MMF CPTAC external eval seed=${SEED} fold=${FOLD} Start: $(date) (job ${SLURM_JOB_ID}) ==="
python -u eval_external.py \
    --seed "${SEED}" --fold "${FOLD}" \
    --results-dir results_n110_mmf --which-splits 5foldcv_n110 \
    --tcga-csv datasets_csv/tcga_paad_all_clean.csv.zip \
    --tcga-split-dir "splits/5foldcv_n110/tcga_paad_seed${SEED}" \
    --cptac-csv datasets_csv/cptac_paad_n110_external.csv.zip \
    --tcga-data-root /pub/wonseukl/Path-ViT/porpoise/data_root_true_resnet50 \
    --cptac-data-root /pub/wonseukl/Path-ViT/data/porpoise_style_features/cptac \
    --out-csv "${OUT_CSV}"
echo "=== PORPOISE N110 MMF CPTAC external eval seed=${SEED} fold=${FOLD} Complete: $(date) ==="
