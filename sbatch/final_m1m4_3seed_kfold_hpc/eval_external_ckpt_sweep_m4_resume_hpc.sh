#!/bin/bash
#SBATCH --job-name=PVT-eval-ext-sweep-m4-resume
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/final_eval_external_ckpt_sweep_m4_resume.log

# eval_external_ckpt_sweep_hpc.sh(전체 CONFIGS 10개, --time=03:00:00)가 2026-08-30에 M2_final/
# M4_noaux/M4_nodisp 추가로 작업량이 늘면서 M3_hybrid 중간(seed126/fold4)에서 시간 초과로
# 잘렸다 — M4_baseline, M4_hybrid, M4_noaux, M4_nodisp 네 개가 아예 시작도 못 됨. 전체를 다시
# 도는 대신 scripts/final_eval_external_ckpt_sweep.py --only로 빠진 4개만 골라 재실행한다
# (재학습 없음, 순수 forward eval이라 4개 x 2seed(noaux/nodisp) 또는 x3seed(baseline/hybrid)
# x5fold 정도면 2시간이면 충분).
#
# 완료 후 CSV 확인:
#   ls .logs/external_preds/ | grep -E "PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD_(FOLD|XMLP)|PMA_uni2native_INT1500_SS_STG_R_DISP_COX_ADD_FOLD|PMA_uni2native_INT1500_SS_AUX_STG_R_COX_ADD_FOLD" | wc -l
#
# 제출: sbatch sbatch/final_m1m4_3seed_kfold_hpc/eval_external_ckpt_sweep_m4_resume_hpc.sh

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

echo "=== eval-external-ckpt sweep (M4 resume) Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.final_eval_external_ckpt_sweep --only M4_baseline,M4_hybrid,M4_noaux,M4_nodisp
echo "=== eval-external-ckpt sweep (M4 resume) Complete: $(date) ==="
