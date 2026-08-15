#!/bin/bash
#SBATCH --job-name=PVT-M2-coordcat-seed42-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m2_coordcat_seed42_kfold_array_%a.log

# 2026-08-15: M1(WSI 단독)에서 학습 파라미터 없는 SpatialPositionEmbedding을 attn_pool 직전
# patch_tokens에 concat+fusion(Linear->LayerNorm->GELU)으로 주입하는 --coord-embed
# --coord-embed-concat이 5-fold pooled internal 0.4541->0.5352, external 0.5153->0.5254로
# 개선(WSI-only 모달리티 순수성 유지, modality leak 없음)된 것을 M2/M3/M4에도 동일 적용해
# 사다리 전체(m2_novit_seed42/m3_novit_seed42/m4_novit_multiseed의 seed42) 대비 이득이
# 유지되는지 확인한다. 그 외 레시피는 m2_novit_seed42_kfold_array_hpc.sh와 완전히 동일 —
# --coord-embed --coord-embed-concat만 추가.
#
# 완료 후(5개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 5개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M2_uni2_STG_R_DISP_COX_ADD_NOVIT_COORD_CAT --seeds 42 --n-folds 5 --bootstrap 2000
#
# 비교 기준(coord-embed 없는 M2-NOVIT, seed42 5-fold pooled): internal/external은
# m2_novit_seed42_kfold_array_hpc.sh 결과 참조.
#
# 제출: sbatch sbatch/m2_coordcat_seed42_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=42
N_FOLDS=5
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_M2_uni2_STG_R_DISP_COX_ADD_NOVIT_COORD_CAT_kfold5_fold${FOLD}.log"

echo "=== M2+coord-embed-concat(uni2,skip-patch-vit,DISP,cox_add) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M2 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --attn-dispersion \
    --skip-patch-vit \
    --coord-embed --coord-embed-concat \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m2_coordcat_seed42_kfold5_array 2>&1 | tee "${log}"
echo "=== M2+coord-embed-concat(uni2,skip-patch-vit,DISP,cox_add) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
