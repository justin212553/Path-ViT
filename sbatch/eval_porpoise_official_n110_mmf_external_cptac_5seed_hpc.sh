#!/bin/bash
#SBATCH --job-name=PVT-porpoise-off110-eval
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-24
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/eval_porpoise_official_n110_mmf_external_cptac_5seed_array_%a.log

# 2026-10-11: run_porpoise_official_n110_mmf_5seed_array_hpc.sh(results_official_n110_mmf, 공식 입력)의 CPTAC 평가.
# CPTAC 입력은 2026-09-14 prepare_porpoise_cptac_official_geneset.py 산출물(공식 RNA 1543/1553 실측, CNV 82개
# neutral, mutation 4/14 실측 나머지 0 — 공식 입력이 CPTAC에 없어 생기는 불가피한 근사, Methods에 명시).
# scaler는 그 fold의 TCGA train split으로 재구성. array index = seed_idx*5 + fold.

cd /pub/wonseukl/Path-ViT/porpoise
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

SEEDS=(84 126 42 168 210)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((IDX / N_FOLDS))]}
FOLD=$((IDX % N_FOLDS))
OUT_CSV="/pub/wonseukl/Path-ViT/.logs/external_preds/cptac_PORPOISE_OFFICIAL_N110_MMF_seed${SEED}_fold${FOLD}of${N_FOLDS}.csv"

echo "=== PORPOISE OFFICIAL N110 MMF CPTAC external eval seed=${SEED} fold=${FOLD} Start: $(date) (job ${SLURM_JOB_ID}) ==="
python -u eval_external.py \
    --seed "${SEED}" --fold "${FOLD}" \
    --results-dir results_official_n110_mmf --which-splits 5foldcv_n110official \
    --tcga-csv datasets_csv_mutsig/tcga_paad_all_clean.csv.zip \
    --tcga-split-dir "splits/5foldcv_n110official/tcga_paad_seed${SEED}" \
    --cptac-csv datasets_csv_mutsig/cptac_paad_external_clean.csv.zip \
    --tcga-data-root /pub/wonseukl/Path-ViT/porpoise/data_root_true_resnet50 \
    --cptac-data-root /pub/wonseukl/Path-ViT/data/porpoise_style_features/cptac \
    --out-csv "${OUT_CSV}" || exit 1
echo "=== PORPOISE OFFICIAL N110 MMF CPTAC external eval seed=${SEED} fold=${FOLD} Complete: $(date) ==="
