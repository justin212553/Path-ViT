#!/bin/bash
#SBATCH --job-name=PVT-mmp-brca-proto
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/run_mmp_brca_prototype_array_%a.log

# 2026-09-14: MMP(Song et al., ICML 2024, mahmoodlab/MMP) BRCA 재현, 1단계(prototype 학습).
# MMP의 PANTHER aggregation은 학습 전 fold의 train split에서 k-means 기반 "prototype"을
# 미리 fit해서 써야 한다(공식 관례, src/scripts/prototype/clustering.sh). 이 연구는 MMP 공식
# 5-fold(TCGA_BRCA_overall_survival_k=0..4)가 아니라 이 연구 자체 BRCA 프로토콜(seed 84/126 x
# 5-fold + institution external holdout)로 split을 다시 만들었으므로(scripts/prepare_mmp_
# brca_data.py), prototype도 그 10개 split(seed x fold) 각각에 대해 따로 fit해야 한다 —
# 그 세트 안 train case만으로 fit해야 leakage가 없다.
#
# 선행 조건: python -m scripts.prepare_mmp_brca_data --mmp-root <MMP_ROOT>/src \
#            --mmp-dataroot <DATAROOT> 완료 (h5 feature, split csv 전부 생성 확인).
#
# array index 0~9 -> seed_idx = id/5, fold = id%5 (이 프로젝트 전역 관례와 동일).
#
# 제출: sbatch sbatch/run_mmp_brca_prototype_array_hpc.sh

MMP_ROOT="/pub/wonseukl/Path-ViT/mmp/src"
MMP_DATAROOT="/pub/wonseukl/Path-ViT/data/mmp_brca_dataroot"

cd "${MMP_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

SEEDS=(84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

SPLIT_DIR="survival/TCGA_BRCA_k${FOLD}_seed${SEED}"
FEAT='extracted-vit_large_patch16_224.dinov2.uni_mass100k'

echo "=== MMP BRCA prototype fit seed=${SEED} fold=${FOLD} Start: $(date) (job ${SLURM_JOB_ID}) ==="
python -u -m training.main_prototype \
    --mode faiss \
    --data_source "${MMP_DATAROOT}/extracted_mag20x_patch256_fp/${FEAT}/feats_h5" \
    --split_dir "${SPLIT_DIR}" \
    --split_names train \
    --in_dim 1024 \
    --n_proto_patches 100000 \
    --n_proto 16 \
    --n_init 3 \
    --seed "${SEED}"
echo "=== MMP BRCA prototype fit seed=${SEED} fold=${FOLD} Complete: $(date) ==="
