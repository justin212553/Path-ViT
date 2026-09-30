#!/bin/bash
#SBATCH --job-name=PVT-FS-BRCA
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-24
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/fs_brca_%x_%a.log

# 2026-09-29: 리뷰 대응 재실행 배치(BRCA) — paper/TODO_reviewer_feedback.md 1~2절.
# 모든 모델 공통:
#   --external-tss instcv   기관 단위 교차검증(결정 C) — fold마다 outcome 미참조 기관 그룹 하나가 external
#   --fold-safe             RNA z-score, clinical 통계, ClusterPool centroid를 fold train만으로
#   M3/M4 계열은 --rna-aux-weight 0 (논문 레시피 noaux, AUX≈noaux 확인됨)
#   5 seeds x 5 folds = 25 array task
# 모델은 MODEL 환경변수로 — sbatch/submit_fs_batch.sh로 제출할 것.
#
# 선행 조건: data/rna_brca_raw_log2.csv가 HPC에 있어야 함.
# 완료 확인: python scripts/check_pred_completeness.py --tag _INSTCV_FS

cd /pub/wonseukl/Path-ViT/
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84 126 42 168 210)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((IDX / N_FOLDS))]}
FOLD=$((IDX % N_FOLDS))

COMMON="--seed ${SEED} --fold ${FOLD} --n-folds ${N_FOLDS} --external-tss instcv --fold-safe \
  --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 --group-ts 0929fs_brca_${MODEL}"
PMA="scripts.train_brca_m4 --gene-selection consistency --cluster-pool --rna-aux-weight 0"

case "$MODEL" in
  M1)    CMD="scripts.train_brca_m1_m2" ;;
  M2)    CMD="scripts.train_brca_m1_m2 --clinical --clinical-staging --clinical-lr-mult 100" ;;
  M3)    CMD="$PMA --no-clinical" ;;
  M4)    CMD="$PMA --clinical-staging --clinical-lr-mult 100" ;;
  M5)    CMD="scripts.train_brca_m5 --clinical-staging" ;;
  M6)    CMD="scripts.train_brca_m6 --gene-selection consistency" ;;
  M7)    CMD="scripts.train_brca_m7 --gene-selection consistency --clinical-staging --clinical-lr-mult 100" ;;
  M3SA)  CMD="$PMA --self-attn-fusion --no-clinical" ;;
  M4SA)  CMD="$PMA --self-attn-fusion --clinical-staging --clinical-lr-mult 100" ;;
  *) echo "unknown MODEL=$MODEL"; exit 1 ;;
esac

echo "=== FS BRCA ${MODEL} seed=${SEED} fold=${FOLD} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
echo "python -u -m ${CMD} ${COMMON}"
python -u -m ${CMD} ${COMMON}
echo "=== FS BRCA ${MODEL} seed=${SEED} fold=${FOLD} Complete: $(date) ==="
