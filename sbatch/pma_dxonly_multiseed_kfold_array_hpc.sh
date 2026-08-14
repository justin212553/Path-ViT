#!/bin/bash
#SBATCH --job-name=PVT-PMA-dxonly-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_dxonly_multiseed_kfold_array_%a.log

# pma_uni2_coxadd_stg_multiseed_kfold_array_hpc.sh(기존 baseline)에 --dx-only-slides만 추가한
# 대조실험. uni2official 조사(2026-08-14)에서 "official 피처는 DX(진단용/영구절편) 슬라이드만
# 포함해 환자당 평균 슬라이드 수가 확 줄었다"는 것과 "좌표 스케일 버그(attn_dispersion 오염)"
# 두 confound가 동시에 섞여 있어, DX-only 자체의 효과를 그 실험만으로는 분리할 수 없었다.
# 이번엔 좌표 스케일 버그 없이(자체 추출 좌표 그대로, backbone은 baseline과 동일 uni2) DX-only
# 슬라이드 필터링(data/dataset.py::_dx_only_slides, TS/BS 등 냉동절편만 제외, 케이스당 남은 DX는
# 전부 유지)의 효과만 검증한다. 로컬 검증: TCGA 슬라이드/케이스 2.48 -> 1.18로 줄어들고 케이스
# 손실은 없음(152/152 유지, DX가 하나도 없는 케이스는 전체 슬라이드로 폴백).
#
# 주의: --dx-only-slides를 켜면 케이스당 슬라이드 구성 자체가 달라지지만(fold 배정 자체는
# stratify key와 무관하므로 바뀌지 않음 — case_id 집합/fold 멤버십은 baseline과 동일, 슬라이드
# "내용"만 달라짐), risk 계산에 들어가는 정보량이 줄어드니 baseline과 직접 비교 가능하다.
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2_INT1500_SS_AUX_STG_R_DXONLY_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_dxonly_multiseed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PMA_uni2_INT1500_SS_AUX_STG_R_DXONLY_DISP_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== PMA(uni2,cox_add,STG+R,dx-only-slides) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --dx-only-slides \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814pma_dxonly_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA(uni2,cox_add,STG+R,dx-only-slides) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
