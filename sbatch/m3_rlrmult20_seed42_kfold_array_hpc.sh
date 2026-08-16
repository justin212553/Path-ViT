#!/bin/bash
#SBATCH --job-name=PVT-M3-rlrmult20-seed42-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m3_rlrmult20_seed42_kfold_array_%a.log

# 2026-08-15: M3(=M4-NOVIT minus clinical, WSI+RNA)에 --rna-lr-mult 20.0 — rna_encoder param
# group에만 base lr의 20배를 준다. M2의 clinical-lr-mult 성공(fold1: internal 0.4433->0.5355,
# external 0.5531->0.5660)을 RNA 브랜치에도 적용한 것 — fold1 파일럿(로컬): internal
# 0.6188->0.6844, external 0.5841->0.6182 확인.
#
# 완료 후(5개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 5개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M4_uni2_INT1500_SS_AUX_NOCLINICAL_DISP_NOVIT_RLR20 --seeds 42 --n-folds 5 --bootstrap 2000
#
# 비교 기준(lr-mult 없는 M3 baseline, seed42 fold1 단독): internal=0.6188, external=0.5841
#
# 제출: sbatch sbatch/m3_rlrmult20_seed42_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=42
N_FOLDS=5
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_M4_uni2_INT1500_SS_AUX_NOCLINICAL_DISP_NOVIT_RLR20_kfold5_fold${FOLD}.log"

echo "=== M3(=M4-NOVIT minus clinical, uni2, rna-lr-mult20) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4 --no-clinical --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --skip-patch-vit \
    --rna-lr-mult 20.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m3_rlrmult20_seed42_kfold5_array 2>&1 | tee "${log}"
echo "=== M3(=M4-NOVIT minus clinical, uni2, rna-lr-mult20) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
