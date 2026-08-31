#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M4-internal
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_brca_m4_internal_%j.log

# TCGA-BRCA 전체(1058 case)를 institution split 없이 표준 6:2:2 internal로 M4(ViT_PMA)를
# 학습/평가한다 — 2026-08-31, institution(BH) external holdout 결과가 internal/external 둘 다
# 인구 구성이 너무 달라(event rate 3배 차이) 신뢰하기 어렵다고 판단, 우선 internal부터 다시
# 보기로 함(사용자 결정). --external-tss none으로 scripts/brca_common.py의 institution split을
# 끄고 전체 1058명을 그대로 6:2:2에 쓴다.
#
# RNA 유전자 리스트는 재선택 안 함(기존 data/brca_rna_gene_selection/selected_genes_top_1500.csv
# 그대로 재사용, 사용자 지시). single seed, single run — BRCA는 데이터셋이 커서 여러 번 안 돌림
# (사용자 지시).
#
# 완료 후 CSV: .logs/kfold_preds/brca_BRCA_PMA_TOP1500_SS_AUX_seed{SEED}.csv
# (external 없음 — external_loader가 None이라 external 평가/CSV 저장 블록 자체가 안 돈다)
#
# 제출: sbatch sbatch/train_brca_m4_internal_hpc.sh
# (seed를 바꾸고 싶으면 아래 SEED= 한 줄만 수정)

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=84

echo "=== BRCA M4(internal only, 전체 1058명) seed=${SEED} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --external-tss none --group-ts 0831_brca_m4_internal_single
echo "=== BRCA M4(internal only) seed=${SEED} Complete: $(date) ==="
