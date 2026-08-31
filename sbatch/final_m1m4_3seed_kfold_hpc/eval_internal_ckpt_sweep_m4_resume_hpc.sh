#!/bin/bash
#SBATCH --job-name=PVT-eval-int-sweep-m4-resume
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/final_eval_internal_ckpt_sweep_m4_resume.log

# 2026-08-30: M4_noaux/M4_nodisp의 internal kfold_preds CSV를 실수로 지운 사고 복구용.
# train.py --eval-internal-ckpt(--eval-external-ckpt와 동일 관례, 신규 추가)로 이미 저장된
# 체크포인트를 다시 읽어 internal held-out fold 예측만 재추출한다(재학습 없음). 로컬에서
# 스모크테스트(1epoch 체크포인트로 원래 CSV와 byte 단위로 diff 없이 동일하게 복구됨 확인함).
#
# scripts/final_eval_internal_ckpt_sweep.py --only M4_noaux,M4_nodisp 로 이 두 모델만 재추출
# (2seed x 5fold x 2모델 = 20개, 체크포인트당 수십 초 수준이라 2시간이면 충분).
#
# 완료 후 CSV 개수 확인(20개 기대, 각 파일당 하나 — FINALEPOCH 버전은 안 만들어짐, 정상 학습
# 경로에서만 생기는 별개 파일이라 애초에 지워지지 않았을 것):
#   ls .logs/kfold_preds/ | grep -E "PMA_uni2native_INT1500_SS_STG_R_DISP_COX_ADD_FOLD|PMA_uni2native_INT1500_SS_AUX_STG_R_COX_ADD_FOLD" | grep -v FINALEPOCH | wc -l
#
# 완료 후 pooling:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PMA_uni2native_INT1500_SS_STG_R_DISP_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PMA_uni2native_INT1500_SS_AUX_STG_R_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/final_m1m4_3seed_kfold_hpc/eval_internal_ckpt_sweep_m4_resume_hpc.sh

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

echo "=== eval-internal-ckpt sweep (M4 resume) Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.final_eval_internal_ckpt_sweep --only M4_noaux,M4_nodisp
echo "=== eval-internal-ckpt sweep (M4 resume) Complete: $(date) ==="
