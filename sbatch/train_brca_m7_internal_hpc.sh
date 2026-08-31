#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M7-internal
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/train_brca_m7_internal_%j.log

# TCGA-BRCA 전체(1058 case)를 institution split 없이 표준 6:2:2 internal로 M7(ClinicalRNAOnly,
# WSI 없음)을 학습/평가한다 — train_brca_m4_internal_hpc.sh와 동일한 이유/동일 seed로 짝을
# 맞춘다("같은 환경일 때 M7을 넘냐 안 넘냐가 문제", scripts/train_brca_m7.py 참조). WSI가 없어
# 가벼운 모델이라(로컬 실행 시 100epoch 풀 레시피도 1분 내외) 메모리/시간 예산은 M4보다 훨씬
# 작게 잡았다.
#
# RNA 유전자 리스트는 재선택 안 함(기존 selected_genes_top_1500.csv 재사용, 사용자 지시).
# single seed, single run.
#
# 완료 후 CSV: .logs/kfold_preds/brca_BRCA_M7_TOP1500_seed{SEED}.csv (external 없음)
#
# 제출: sbatch sbatch/train_brca_m7_internal_hpc.sh
# (seed를 바꾸고 싶으면 아래 SEED= 한 줄만 수정 — M4와 반드시 동일 seed로 맞출 것)

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=84

echo "=== BRCA M7(internal only, 전체 1058명) seed=${SEED} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m7 --seed "${SEED}" --external-tss none --group-ts 0831_brca_m7_internal_single
echo "=== BRCA M7(internal only) seed=${SEED} Complete: $(date) ==="
