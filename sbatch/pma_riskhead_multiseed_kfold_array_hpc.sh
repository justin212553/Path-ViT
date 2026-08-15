#!/bin/bash
#SBATCH --job-name=PVT-PMA-riskhead-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_riskhead_multiseed_kfold_array_%a.log

# pma_uni2_coxadd_stg_multiseed_kfold_array_hpc.sh(기존 baseline)에 --tile-risk-head만 추가한
# 대조실험. diagnose_pma_wsi_structure.py 실측(2026-08-14) — MultiComponentPooling의 "top"
# 컴포넌트가 attn_weights(patch attention entropy~0.999, 사실상 uniform으로 붕괴)로 top-k를
# 선정하고 있어 독립적인 관점이 아니었다. 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology)
# MorphologyBurdenPooling(scripts/models/morphology_burden_mil.py)을 참고해:
#   1) top-k 선정을 self.attn과 파라미터를 공유하지 않는 별도의 단순 TileRiskHead(게이트 없는
#      얕은 MLP)로 분리(models/multi_component_pooling.py::TileRiskHead) — 이 헤드는 attn_pool의
#      붕괴를 물려받을 이유가 없다.
#   2) 레퍼런스의 risk_stats(패치별 risk 점수 분포를 요약하는 10개 스칼라 — mean/std/max/
#      quantile(25/50/75)/top05/top10/frac_over_50/frac_over_70)를 risk_head 입력에
#      spatial_feat과 나란히 추가.
#
# fold0/seed42 로컬 파일럿(2026-08-14): internal test_c_index 0.4851(baseline)->0.6474(+0.16),
# external 0.5912->0.5758(-0.015) — internal이 크게 뛰었지만 fold0/단일시드라 신뢰하지 않고
# 전체 스케일로 검증한다(이번 세션에 uni2official/tumor_type_embed/complete-24m/M4-NOVIT 등
# fold0 파일럿이 좋아 보였다가 전체 스케일에서 사라지거나 뒤집힌 전례가 이미 여러 번 있었음).
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_RISKHEAD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_riskhead_multiseed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_RISKHEAD_kfold5_fold${FOLD}.log"

echo "=== PMA(uni2,cox_add,STG+R,tile-risk-head) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --tile-risk-head \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814pma_riskhead_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA(uni2,cox_add,STG+R,tile-risk-head) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
