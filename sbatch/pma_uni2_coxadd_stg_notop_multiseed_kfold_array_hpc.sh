#!/bin/bash
#SBATCH --job-name=PVT-PMA-uni2-notop-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_uni2_coxadd_stg_notop_multiseed_kfold_array_%a.log

# pma_uni2_coxadd_stg_multiseed_kfold_array_hpc.sh(원래 4개 관점 다 쓰는 baseline)의
# top-k 성분 제거 버전. 2026-08-09 seed126 단일 split 파일럿에서:
#   baseline(4개):        internal=0.5311  external=0.6372
#   top-k 제거(3개):      internal=0.5456(+0.0145)  external=0.6384(+0.0012)
#   top25%로 확장(4개):   internal=0.5353(+0.0042)  external=0.6391(+0.0019)
# 중 top-k 완전 제거가 internal/external 둘 다 가장 크게 개선됐다 — mean/std/attn을
# 각각 하나씩 지운 버전과 사후 zero-ablation 진단(scripts/diagnose_pma_component_reliance.py)
# 둘 다 같은 결론을 가리켰다. 다만 seed126 하나·single-split 하나뿐인 결과라 재현되는지
# 3seed(42/84/126) x 5-fold로 검증한다. --drop-component top만 추가하고 나머지는
# pma_uni2_coxadd_stg_multiseed_kfold_array_hpc.sh와 완전히 동일한 레시피/seed-fold 매핑
# (IDX -> seed_idx=IDX/5, fold=IDX%5).
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOTOP --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_uni2_coxadd_stg_notop_multiseed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(42 84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOTOP_kfold5_fold${FOLD}.log"

echo "=== PMA(uni2,cox_add,STG+R,NOTOP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add --drop-component top \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0809pma_uni2_coxadd_stg_notop_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA(uni2,cox_add,STG+R,NOTOP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
