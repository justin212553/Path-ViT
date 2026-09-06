#!/bin/bash
#SBATCH --job-name=PVT-M7-INT1500-STG-R-COXADD
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m7_int1500_stg_r_coxadd_kfold_array_%a.log

# 2026-09-06: M7(WSI 없음, Clinical+RNAseq, models/clinical_rna_only.py::ClinicalRNAOnly)
# 기준선(paper/results_table_pma_family_3seed_kfold_ci.md, 태그 M7_INT1500_STG_R_COX_ADD,
# internal C=0.6552/external C=0.6221)의 .logs/kfold_preds·external_preds CSV가 이번 HPC
# 계정/디렉토리엔 없어서(예전 세션 산출물, 로컬로만 옮겨졌거나 정리됨) paired bootstrap
# 비교(scripts/paired_bootstrap_delta.py) 입력이 없다 — 재학습해서 다시 만든다.
#
# 이 태그는 --M7의 "legacy" 조합(RNA 인코더 교체+clinical raw 직결)이 2026-08-21에 코드
# 자체의 영구 기본 동작으로 원복됐으므로(paper/results_table_pma_family_3seed_kfold_ci.md 상단
# 기록 참조 — 당시 쓰던 --legacy-rna-encoder/--legacy-clinical-coxadd 플래그는 이미 삭제됨),
# **지금 코드베이스에서 --M7을 그냥 표준으로 돌리면 그대로 이 태그가 재현된다** — 특별한
# ablation 플래그 불필요.
#
# train.py가 아니라 train_light.py를 쓴다 — --M7/--M5/--M6 계열은 train_light.py 전용(WSI가
# 아예 없는 모델이라 train.py의 WSI 관련 인자/파이프라인이 필요 없음). WSI가 없어 backbone
# 인자 자체가 없다(태그에도 _uni2native 같은 backbone 접미사가 없는 이유).
#
# 태그 구성 확인(train_light.py model_prefix 체인 직접 대조):
#   M7(base) -> _INT1500(--rna-genes literature_1500_intersection) -> _STG(--clinical-staging)
#   -> _R(--clinical-margin) -> _COX_ADD(--combine-mode cox_add, 기본 concat 아님)
#   = M7_INT1500_STG_R_COX_ADD (정확히 일치)
#
# 2seed(84,126) x 5fold — 이 프로젝트 표준(seed42는 WSI 포함 모델에서 유독 튀어 최종 집계에서
# 제외해온 관례이지만, M7 자체는 WSI가 없어도 기존 확정 수치와의 일관성을 위해 동일하게 2seed만).
# sbatch/porpoise_no_aux_multiseed_kfold_array_hpc.sh와 동일한 SEED_IDX/FOLD 인덱싱 관례.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M7_INT1500_STG_R_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model M7_INT1500_STG_R_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000
# (재현되면 기존 문서의 internal=0.6552/external=0.6221과 비교해 확인할 것 — 다르면 코드베이스가
# 그 사이 더 바뀌었다는 뜻이므로 바로 이어서 paired bootstrap을 돌리기 전에 먼저 보고할 것.)
#
# 제출: sbatch sbatch/m7_int1500_stg_r_coxadd_multiseed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_M7_INT1500_STG_R_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== M7(INT1500,STG,R,COX_ADD) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train_light.py --M7 --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906m7_int1500_stg_r_coxadd_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== M7(INT1500,STG,R,COX_ADD) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
