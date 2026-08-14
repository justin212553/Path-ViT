#!/bin/bash
#SBATCH --job-name=PVT-PMA-refcohort-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_refcohort_multiseed_kfold_array_%a.log

# pma_uni2_coxadd_stg_multiseed_kfold_array_hpc.sh(기존 baseline)에 --reference-cohort만 추가한
# 대조실험. 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology)는 실제 M1-M4/M7 학습 시 "24개월
# 시점 생존 여부 확정(complete-24m)" case만 쓴다(data/reference_cohort.py::
# reference_eligible_case_ids, TCGA 160->112, CPTAC 140->93, 총 205명 — 2026-08-14 조사).
# 우리는 이 필터를 기본으로 안 쓰는데, 조기 censoring이 많이 섞인 게 WSI 신호를 흐리는지
# 확인하는 대조실험. --dx-only-slides(DX-only 슬라이드 필터)와 함께 이번 조사 묶음이다 — 둘 다
# 각자 따로(단일 요인) 3seed x 5fold로 돌려서 baseline과 비교한다.
#
# fold0/seed42 로컬 파일럿(2026-08-14): internal test c-index 0.485->0.548(+0.063, N=31->20,
# 노이즈 큼 — 이전에 uni2official/tumor_type_embed도 fold0 파일럿에서 긍정적이었다가 3seed x
# 5fold 전체에서 뒤집힌 전례 있음), external c-index 0.591->0.592(사실상 flat, N=144->125).
# 파일럿만으론 결론 낼 수 없어 전체 스케일로 검증한다.
#
# 주의: --reference-cohort는 case_id 집합 자체를 줄이므로(TCGA 152->약 112, 즉 훨씬 작은 표본),
# 같은 seed라도 fold 배정과 train/val/test 크기 자체가 baseline과 완전히 다르다 — c-index
# 직접 비교보다 "패턴(내부/외부 방향, 과적합 정도)"을 보는 데 무게를 둘 것. --external을 함께
# 쓰므로 external(CPTAC) 평가 코호트도 144명이 아니라 93명 안쪽으로 줄어든다(레퍼런스가 CPTAC
# 평가도 같은 205명 풀 안에서 하는 것과 동일 관례) — baseline external(144명 전체)과 직접
# 비교하려면 이 표본 차이를 감안할 것.
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2_INT1500_SS_AUX_STG_R_REFCOHORT_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_refcohort_multiseed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PMA_uni2_INT1500_SS_AUX_STG_R_REFCOHORT_DISP_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== PMA(uni2,cox_add,STG+R,reference-cohort) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --reference-cohort \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814pma_refcohort_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA(uni2,cox_add,STG+R,reference-cohort) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
