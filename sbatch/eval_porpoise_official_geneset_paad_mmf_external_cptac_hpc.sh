#!/bin/bash
#SBATCH --job-name=PVT-porpoise-officialgene-eval-cptac
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/eval_porpoise_official_geneset_paad_mmf_external_cptac_array_%a.log

# 2026-09-14: run_porpoise_official_geneset_paad_mmf_2seed_array_hpc.sh(results_official_geneset_mmf,
# 공식 datasets_csv_mutsig 유전자 세트로 재학습한 진짜 official 재현)의 CPTAC-PDAC external 평가.
#
# 선행 조건:
#   1) run_porpoise_official_geneset_paad_mmf_2seed_array_hpc.sh 완료 (5 checkpoint x 2 seed)
#   2) python -m scripts.prepare_porpoise_cptac_official_geneset 완료
#      (porpoise/datasets_csv_mutsig/cptac_paad_external_clean.csv.zip 생성 확인)
#
# 이 CSV는 RNA-seq 1543/1553 실측 + CNV 82개 전부 neutral 근사 + mutation 4/14 실측 근사를
# 포함한다(정확한 근거는 scripts/prepare_porpoise_cptac_official_geneset.py 상단 주석 참조).
# 이 근사들은 논문 Methods/Limitations에 명시해야 한다.
#
# 완료 후 pooling:
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PORPOISE_OFFICIALGENE_MMF --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출(선행 조건 확인 후): sbatch sbatch/eval_porpoise_official_geneset_paad_mmf_external_cptac_hpc.sh

cd /pub/wonseukl/Path-ViT/porpoise

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

SEEDS=(84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

OUT_CSV="/pub/wonseukl/Path-ViT/.logs/external_preds/cptac_PORPOISE_OFFICIALGENE_MMF_seed${SEED}_fold${FOLD}of${N_FOLDS}.csv"

echo "=== PORPOISE 공식 유전자 세트 MMF CPTAC external eval seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u eval_external.py \
    --seed "${SEED}" --fold "${FOLD}" \
    --results-dir results_official_geneset_mmf \
    --tcga-csv datasets_csv_mutsig/tcga_paad_all_clean.csv.zip \
    --tcga-split-dir splits/5foldcv/tcga_paad \
    --cptac-csv datasets_csv_mutsig/cptac_paad_external_clean.csv.zip \
    --tcga-data-root /pub/wonseukl/Path-ViT/porpoise/data_root_true_resnet50 \
    --cptac-data-root /pub/wonseukl/Path-ViT/data/porpoise_style_features/cptac \
    --out-csv "${OUT_CSV}"
echo "=== PORPOISE 공식 유전자 세트 MMF CPTAC external eval seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
