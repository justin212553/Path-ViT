#!/bin/bash
#SBATCH --job-name=PVT-M4-lit1500-uni2native
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m4_literature1500_cnv_mut_clr100_uni2native_kfold_array_%a.log

# 2026-09-05: m4_pdaccons1500_cnv_mut_clr100_uni2native_kfold_array_hpc.sh와 동일 레시피,
# RNA 패널만 literature_1500(문헌 큐레이션, cohort 라벨 일부 참조 — pdac_consistency_1500
# 대비 "leaky"하다고 이미 플래그됨, project_rna_gene_selection_leakage 메모 참조)으로 교체.
# --backbone uni2native 명시(오늘 세션 전체의 backbone 누락 사고 재발 방지).
#
# 로컬 단일 fold 참고(오늘 세션 중 CLR100 스윕 기록, 2seed x 5fold pooled):
#   literature_1500(leaky)+CLR100: internal 0.639 (p=0.013), external 0.651 (p<0.0001)
#   — pdac_consistency_1500+CLR100(objective, 위 스크립트): internal 0.600, external 0.639
#   leaky 패널이 숫자는 더 높지만 리키지 의심으로 헤드라인으로 안 쓰기로 함 — 이 재현은
#   "같은 backbone/레시피로 둘 다 정식 재현해서 나란히 보고 싶다"(사용자 요청)는 목적.
#
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5. seed=84/126, fold=0..4.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M4_uni2native_EXT1500_CNV_STG_R_MUT_COX_ADD_CLR100_LRMW10 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   (--rna-genes literature_1500의 실제 model_prefix 태그는 로그 상단 "Model:" 줄에서 확인할 것 —
#    literature_1500이 "_EXT1500"으로 붙는지 "_LIT1500"으로 붙는지는 train.py의 rna_genes 분기
#    문자열에 따라 달라질 수 있음.)
#
# 제출: sbatch sbatch/m4_literature1500_cnv_mut_clr100_uni2native_kfold_array_hpc.sh

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

log=".logs/train_tcga_m4_literature1500_cnv_mut_clr100_uni2native_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M4(literature_1500+CNV+mutation+STG+R+CLR100, uni2native) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native --rna-genes literature_1500 --use-cnv --clinical-mutation \
    --clinical-staging --clinical-margin --combine-mode cox_add \
    --clinical-lr-mult 100 --lr-mult-warmup-epochs 10 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --group-ts 0905_m4_literature1500_cnv_mut_clr100_uni2native_kfold_array 2>&1 | tee "${log}"
echo "=== M4(literature_1500+CNV+mutation+STG+R+CLR100, uni2native) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
