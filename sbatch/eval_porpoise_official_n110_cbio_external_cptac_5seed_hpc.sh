#!/bin/bash
#SBATCH --job-name=PVT-porpoise-off110-cbio-eval
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-24
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/eval_porpoise_official_n110_cbio_external_cptac_5seed_array_%a.log

# 2026-10-11: 공식 입력판 PORPOISE(results_official_n110_mmf, 이미 학습 완료)를 CPTAC "실측" CNV·mutation으로
# 다시 평가 — 재학습 없음. 기존 eval_porpoise_official_n110_mmf_*의 CPTAC 입력은 CNV 82개 neutral, mutation 10/14개
# 0 근사였다. cBioPortal pancreas_cptac_gdc의 GISTIC CNV(79/82 실측)·mutation(14/14, 135/136명)으로 교체한
# CSV를 쓴다. 선행: python scripts/porpoise_cptac_cbio_cnv_mut.py patch (HPC, repo 루트)
# 출력 태그 PORPOISE_OFFICIAL_N110_CBIO_MMF(external만). internal은 같은 모델이라 PORPOISE_OFFICIAL_N110_MMF 그대로.

cd /pub/wonseukl/Path-ViT/porpoise
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

SEEDS=(84 126 42 168 210)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((IDX / N_FOLDS))]}
FOLD=$((IDX % N_FOLDS))
OUT_CSV="/pub/wonseukl/Path-ViT/.logs/external_preds/cptac_PORPOISE_OFFICIAL_N110${QC:+_QC}_CBIO_MMF_seed${SEED}_fold${FOLD}of${N_FOLDS}.csv"

echo "=== PORPOISE OFFICIAL N110 MMF CPTAC(cBioPortal CNV/mut) external eval seed=${SEED} fold=${FOLD} Start: $(date) (job ${SLURM_JOB_ID}) ==="
python -u eval_external.py \
    --seed "${SEED}" --fold "${FOLD}" \
    --results-dir results_official_n110_mmf${QC:+_qc} --which-splits 5foldcv_n110official${QC:+qc} \
    --tcga-csv datasets_csv_mutsig/tcga_paad_all_clean.csv.zip \
    --tcga-split-dir "splits/5foldcv_n110official${QC:+qc}/tcga_paad_seed${SEED}" \
    --cptac-csv datasets_csv_mutsig/cptac_paad_external_cbio.csv.zip \
    --tcga-data-root /pub/wonseukl/Path-ViT/porpoise/data_root_true_resnet50 \
    --cptac-data-root /pub/wonseukl/Path-ViT/data/porpoise_style_features/cptac \
    --out-csv "${OUT_CSV}" || exit 1
echo "=== PORPOISE OFFICIAL N110 MMF CPTAC(cBioPortal CNV/mut) external eval seed=${SEED} fold=${FOLD} Complete: $(date) ==="
