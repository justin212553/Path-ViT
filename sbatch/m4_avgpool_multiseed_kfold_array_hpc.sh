#!/bin/bash
#SBATCH --job-name=PVT-M4-avgpool-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m4_avgpool_multiseed_kfold_array_%a.log

# M4_AVGPOOL_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD — WSI pooling 구조 단순화 사다리의 마지막
# 칸(2026-08-14 논의): PMA(4-component+co-attention) -> M4(단일 gated-ABMIL) ->
# M4+skip-patch-vit(patch-mixing까지 제거) -> PMA+tile-risk-head(top-k를 독립 헤드로 분리+
# risk_stats 추가) 전부 external이 baseline과 다를 게 없거나(-0.03까지) 더 나빠졌다.
# diagnose_pma_wsi_structure.py가 실측한 patch attention entropy~0.999(사실상 uniform)를
# 생각하면, "attention을 고치거나 우회하려는" 시도보다 아예 학습 파라미터가 전혀 없는 순수
# 평균 풀링(models/vit_m4_avgpool.py::ViT_M4_AvgPool, train.py --M4 --avgpool)이 더 나을 수도
# 있다는 가설 — 이미 사실상 균일하게 동작하던 게이트 파라미터를 완전히 없애 불필요한 gradient
# 노이즈 자체를 지운다.
#
# fold0/seed42 로컬 파일럿(2026-08-14): M4(attention 포함) internal=0.6965/external=0.5823 vs
# M4+avgpool internal=0.7070/external=0.5798 — 거의 동일(오차 범위). 전체 스케일 검증은 이번이
# 처음이다(파일럿 이후 tile-risk-head 조사로 먼저 넘어갔었음).
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M4_AVGPOOL_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m4_avgpool_multiseed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_M4_AVGPOOL_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== M4+avgpool(uni2,cox_add,STG+R,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4 --avgpool --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814m4_avgpool_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== M4+avgpool(uni2,cox_add,STG+R,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
