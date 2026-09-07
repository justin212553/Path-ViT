#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-pdaccons2000
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_final_pdaccons2000_2seed_kfold_array_%a.log

# 2026-09-06: pdac_consistency_1500(leak-free) 단독이 literature_1500 대비 external에서
# 약해 보이는 문제(둘 다 비유의였지만 점추정치 차이 존재, 0.619 vs 0.647) 개선 시도 1 —
# 우리 코호트 라벨은 전혀 안 건드리고, 같은 외부 소스에서 유전자 수만 1500->2000으로 늘림
# (--rna-genes pdac_consistency_2000, 기존에 이미 지원되던 옵션). 나머지는 확정 최종 레시피와
# 완전히 동일.
#
# 2시드(84,126)만 — "일단 방향성이 있는지"만 빠르게 확인하는 단계(사용자 지시: 5시드는
# 너무 오래 걸리니 2시드로). 방향이 있으면 나중에 5시드로 확장.
#
# 같은 목적의 다른 후보(pdac_consistency_all5_pluslit, 문헌 subtype 유전자 추가)는
# sbatch/porpoise_final_pdacconsall5pluslit_2seed_kfold_array_hpc.sh.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PORPOISE_uni2native_PDACCONS2000_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PORPOISE_uni2native_PDACCONS2000_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/porpoise_final_pdaccons2000_2seed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2native_PDACCONS2000_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1_kfold5_fold${FOLD}.log"

echo "=== PORPOISE final recipe(pdac_consistency_2000) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes pdac_consistency_2000 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906porpoise_final_pdaccons2000_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== PORPOISE final recipe(pdac_consistency_2000) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
