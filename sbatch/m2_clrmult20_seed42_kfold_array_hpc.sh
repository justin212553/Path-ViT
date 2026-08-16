#!/bin/bash
#SBATCH --job-name=PVT-M2-clrmult20-seed42-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m2_clrmult20_seed42_kfold_array_%a.log

# 2026-08-15: M2(WSI+Clinical, concat)에 --clinical-lr-mult 20.0 — clinical_encoder param
# group에만 base lr의 20배를 준다(cfg.train.lr=1e-5 -> 2e-4, LightTrainConfig.lr=1e-3에 근접).
# scripts/diagnose_m2_branch_swap.py 실측: 공동학습된 clinical_encoder가 M5(clinical 단독,
# lr=1e-3) 대비 internal -0.075/external -0.018 떨어지는데 WSI는 M1 대비 거의 안 상함 —
# clinical이 WSI와 같은 lr로 경쟁하면 밀려난다는 가설. fold1 파일럿(로컬): internal
# 0.4433->0.5355, external 0.5531->0.5660 확인.
#
# 완료 후(5개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 5개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M2_uni2_STG_R_DISP_NOVIT_CLR20 --seeds 42 --n-folds 5 --bootstrap 2000
#
# 비교 기준(coord/lr-mult 없는 M2 concat baseline, seed42 fold1 단독): internal=0.4433, external=0.5531
#
# 제출: sbatch sbatch/m2_clrmult20_seed42_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=42
N_FOLDS=5
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_M2_uni2_STG_R_DISP_NOVIT_CLR20_kfold5_fold${FOLD}.log"

echo "=== M2(uni2,concat,clinical-lr-mult20) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M2 --skip-patch-vit --attn-dispersion --clinical-margin --clinical-staging \
    --clinical-lr-mult 20.0 \
    --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m2_clrmult20_seed42_kfold5_array 2>&1 | tee "${log}"
echo "=== M2(uni2,concat,clinical-lr-mult20) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
