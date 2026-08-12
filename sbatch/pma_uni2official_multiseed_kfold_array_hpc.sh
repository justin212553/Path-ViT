#!/bin/bash
#SBATCH --job-name=PVT-PMA-uni2official-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_uni2official_multiseed_kfold_array_%a.log

# pma_uni2_coxadd_stg_multiseed_kfold_array_hpc.sh의 backbone만 uni2official로 바꾼 대조 실험.
#
# 2026-08-12: 우리 자체 UNI2-h feature 추출 파이프라인(1024px@1.0MPP -> 512 리사이즈, 실효
# 2.0MPP)이 UNI2-h 공식 학습/검증 스펙(256px@20x, ~0.5MPP)과 4배 어긋난다는 게 확인됐다.
# MahmoodLab이 공식 스펙으로 직접 뽑아 배포한 feature(HuggingFace MahmoodLab/UNI2-h-features)를
# scripts/convert_uni2h_official_features.py로 변환해 우리 파이프라인에 끼워 넣은 게
# --backbone uni2official (data/dataset.py, models/vit_m1.py TILE_ENCODER_REGISTRY 참조).
#
# seed42/fold0 파일럿 결과: internal 0.4851->0.5954(+0.110), external 0.5911->0.6065(+0.015) —
# 같은 seed/fold/레시피에서 feature만 바꿔 나온 결과라 고무적이지만, fold0는 이 코호트에서
# 원래 변동성이 제일 큰 fold(N=31)라 단일 파일럿만으로는 못 믿는다 — 전체 3seed x 5fold로 검증.
#
# [전제] uni2official_features_for_hpc.zip(scripts/zip_uni2official_features_for_hpc.py 산출물)이
# 이미 HPC의 /pub/wonseukl/Path-ViT/에 풀려 있어야 한다(각 슬라이드 디렉토리 아래
# features_uni2official.pt + coords_uni2official.pt). 환자 커버리지는 100%가 아님(TCGA 152명 중
# 150명, CPTAC 159명 중 144명만 공식 feature로 커버됨 — MahmoodLab이 처리한 슬라이드 목록에 없는
# 환자는 이번 실험에서 자동 제외된다).
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2official_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_uni2official_multiseed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PMA_uni2official_INT1500_SS_AUX_STG_R_DISP_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== PMA(uni2official,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2official \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0812pma_uni2official_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA(uni2official,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
