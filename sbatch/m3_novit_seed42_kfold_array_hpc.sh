#!/bin/bash
#SBATCH --job-name=PVT-M3-novit-seed42-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m3_novit_seed42_kfold_array_%a.log

# 2026-08-14: M1~M4 사다리를 M4-NOVIT과 같은 WSI 아키텍처로 통일하는 작업의 M3(WSI+RNA,
# clinical 제외). "M4-NOVIT에서 clinical cox_add만 제거"로 정의(사용자 지시) — RNA-guided
# FiLM ABMIL + skip-patch-vit + SS(patch-keep-frac 0.8) + AUX(rna-aux-weight 1.0) +
# DISP(attn-dispersion)는 그대로 유지하고 clinical만 뺀다. models/vit_m4.py에 use_clinical
# 파라미터를 오늘 새로 추가했다(--no-clinical). cox_add는 clinical이 있어야 의미가 있는
# 결합 방식이라 --no-clinical과 함께 못 쓴다(코드가 직접 막음) — combine-mode는 기본(concat)
# 그대로 둔다(use_clinical=False라 어차피 clinical_encoder 자체가 안 만들어짐).
#
# 완료 후(5개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 5개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M4_uni2_INT1500_SS_AUX_NOCLINICAL_DISP_NOVIT --seeds 42 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m3_novit_seed42_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=42
N_FOLDS=5
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_M4_uni2_INT1500_SS_AUX_NOCLINICAL_DISP_NOVIT_kfold5_fold${FOLD}.log"

echo "=== M3(=M4-NOVIT minus clinical, uni2, SS+AUX+DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4 --no-clinical --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --skip-patch-vit \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814m3_novit_seed42_kfold5_array 2>&1 | tee "${log}"
echo "=== M3(=M4-NOVIT minus clinical, uni2, SS+AUX+DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
