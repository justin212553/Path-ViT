#!/bin/bash
#SBATCH --job-name=PVT-FINAL-eval-ext-sweep
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/final_eval_external_ckpt_sweep.log

# 학습 스크립트(M1, M2/M3/M4 baseline+hybrid, 2026-08-21에 M2_final(selfattn+cox_add),
# 2026-08-30에 M4_noaux/M4_nodisp 추가)가 저장한 체크포인트를 --eval-external-ckpt로 다시 읽어
# external_preds CSV를 생성한다(재학습 없음, 순수 forward eval이라 체크포인트당 수십 초 수준).
# 일반 --fold 학습 경로가 external CSV를 저장하지 않는다는 걸 뒤늦게 발견해서(2026-08-16) 추가한
# 후속 스크립트 — scripts/final_eval_external_ckpt_sweep.py 참조. 이미 CSV가 있는 모델도 다시
# 돌지만 결과는 그대로라 무해.
#
# 2026-08-30: CONFIGS가 7->10개로 늘면서 --time=03:00:00으로는 부족해 M3_hybrid 중간에서
# 시간 초과로 잘리는 사고가 있었음(M4_baseline~M4_nodisp 전부 미실행) — 05:00:00으로 상향.
# 급하게 일부만 다시 돌리고 싶으면 전체를 기다리지 말고
# scripts/final_eval_external_ckpt_sweep.py --only <label1>,<label2>,...로 타겟팅할 것
# (sbatch/final_m1m4_3seed_kfold_hpc/eval_external_ckpt_sweep_m4_resume_hpc.sh가 그 예시).
#
# 완료 후 CSV 개수 확인(모델당 seed*fold, M2_hybrid는 seed126/fold3 학습 자체가 안 됐어서 -1,
# M4_noaux/M4_nodisp는 2seed(84/126)x5fold=10개씩 — 3seed 모델과 개수가 다름):
#   ls .logs/external_preds/ | grep uni2native | wc -l
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
