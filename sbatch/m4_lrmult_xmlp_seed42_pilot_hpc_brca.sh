#!/bin/bash
#SBATCH --job-name=PVT-brca-m4-lrmult
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_lrmult_xmlp_seed42_pilot.log

# scripts/train_brca_m4.py에 이번 PAAD 세션에서 검증된 --clinical-lr-mult/--rna-lr-mult(20x)와
# --wsi-extra-mlp를 이식해 TCGA-BRCA(1058명, single split, uni v1 backbone)로 한 번 돌려본다.
# 배경: paper/notes_wsi_signal_and_seed_variance.md 6절 — 2026-07-22 seed42 단일 결과에서
# BRCA만 유일하게 WSI가 순증분(M7 0.6620 vs M4/PMA 0.7155, +0.0535)을 보였다. 그 결과는
# lr-mult/XMLP 없이 나온 것이라, PAAD에서 branch-competition을 해소해준 이 기법들을 더하면
# BRCA(표본 7배, 노이즈가 훨씬 적을 것으로 기대)에서 얼마나/어떻게 바뀌는지 궁금해서 확인.
# uni2-native로는 아직 안 돌림(BRCA 타일 재추출 필요 — 범위를 줄여 uni v1 그대로 진행).
#
# [사전 준비] data/patches_tcga_brca/tiles/*/features_uni.pt, data/brca_clinical.csv,
# data/brca_slide_manifest.csv, data/brca_rna_gene_selection/selected_genes_top_1500.csv가
# HPC에 있어야 함 — 2026-08-11 pancancer_paad_brca 실험 때 한 번 스테이징된 적 있음
# (scripts/zip_brca_for_hpc.py 산출물 unzip). 없으면 로컬에서 다시 zip해서 옮길 것.
#
# 제출: sbatch sbatch/brca_m4_lrmult_xmlp_seed42_pilot_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

echo "=== BRCA M4(PMA,uni)+CLR20+RLR20+XMLP seed=42 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed 42 \
    --patch-keep-frac 0.8 --rna-aux-weight 1.0 \
    --clinical-lr-mult 20.0 --rna-lr-mult 20.0 --wsi-extra-mlp \
    --group-ts 0816_brca_m4_lrmult_xmlp_pilot
echo "=== BRCA M4(PMA,uni)+CLR20+RLR20+XMLP seed=42 Complete: $(date) ==="
