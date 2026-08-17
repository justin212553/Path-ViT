#!/bin/bash
#SBATCH --job-name=PVT-FINAL-eval-ext-sweep
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/final_eval_external_ckpt_sweep.log

# 7개 학습 스크립트(M1, M2/M3/M4 baseline+hybrid)가 저장한 체크포인트 105개(7 x 3seed x 5fold)를
# --eval-external-ckpt로 다시 읽어 external_preds CSV를 생성한다(재학습 없음, 순수 forward eval
# 이라 체크포인트당 수십 초 수준). 일반 --fold 학습 경로가 external CSV를 저장하지 않는다는 걸
# 뒤늦게 발견해서(2026-08-16) 추가한 후속 스크립트 — scripts/final_eval_external_ckpt_sweep.py
# 참조.
#
# 완료 후 CSV 개수 확인(모델당 15개씩, M2_hybrid는 seed126/fold3 학습 자체가 안 됐어서 14개):
#   ls .logs/external_preds/ | grep uni2native | wc -l   # 104개 기대
#
# 제출: sbatch sbatch/final_m1m4_3seed_kfold_hpc/eval_external_ckpt_sweep_hpc.sh

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

echo "=== FINAL eval-external-ckpt sweep Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.final_eval_external_ckpt_sweep
echo "=== FINAL eval-external-ckpt sweep Complete: $(date) ==="
