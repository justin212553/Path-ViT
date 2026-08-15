#!/bin/bash
#SBATCH --job-name=PVT-M4-coordcat-seed42-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m4_coordcat_seed42_kfold_array_%a.log

# 2026-08-15: M1의 --coord-embed --coord-embed-concat 개선(internal 0.4541->0.5352,
# external 0.5153->0.5254, WSI-only 순수성 유지, modality leak 없음)을 M4-NOVIT(사다리
# 최종 슬롯)에도 적용. 그 외 레시피는 m4_novit_multiseed_kfold_array_hpc.sh(seed42 부분)와
# 완전히 동일 — --coord-embed --coord-embed-concat만 추가.
#
# 비교 기준(coord-embed 없는 M4-NOVIT, seed42 5-fold pooled, ensemble): internal=0.6885,
# external=0.6031 (사다리 재검증용으로 recompute된 clean seed42-only 값).
#
# 완료 후(5개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 5개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT_COORD_CAT --seeds 42 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m4_coordcat_seed42_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=42
N_FOLDS=5
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT_COORD_CAT_kfold5_fold${FOLD}.log"

echo "=== M4+coord-embed-concat(uni2,cox_add,STG+R,DISP,skip-patch-vit) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4 --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --skip-patch-vit \
    --coord-embed --coord-embed-concat \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m4_coordcat_seed42_kfold5_array 2>&1 | tee "${log}"
echo "=== M4+coord-embed-concat(uni2,cox_add,STG+R,DISP,skip-patch-vit) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
