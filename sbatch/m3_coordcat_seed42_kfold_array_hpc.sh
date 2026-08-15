#!/bin/bash
#SBATCH --job-name=PVT-M3-coordcat-seed42-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m3_coordcat_seed42_kfold_array_%a.log

# 2026-08-15: M1의 --coord-embed --coord-embed-concat 개선(internal 0.4541->0.5352,
# external 0.5153->0.5254, WSI-only 순수성 유지)을 M3(=M4-NOVIT minus clinical)에도 적용.
# 그 외 레시피는 m3_novit_seed42_kfold_array_hpc.sh와 완전히 동일.
#
# 완료 후(5개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 5개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M4_uni2_INT1500_SS_AUX_NOCLINICAL_DISP_NOVIT_COORD_CAT --seeds 42 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m3_coordcat_seed42_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=42
N_FOLDS=5
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_M4_uni2_INT1500_SS_AUX_NOCLINICAL_DISP_NOVIT_COORD_CAT_kfold5_fold${FOLD}.log"

echo "=== M3+coord-embed-concat(=M4-NOVIT minus clinical, uni2, SS+AUX+DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4 --no-clinical --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --skip-patch-vit \
    --coord-embed --coord-embed-concat \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m3_coordcat_seed42_kfold5_array 2>&1 | tee "${log}"
echo "=== M3+coord-embed-concat(=M4-NOVIT minus clinical, uni2, SS+AUX+DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
