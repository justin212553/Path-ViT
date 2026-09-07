#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-pdacall5pluslit
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_final_pdacconsall5pluslit_2seed_kfold_array_%a.log

# 2026-09-06: pdac_consistency_1500 단독의 external 약세 개선 시도 2 — 우리 코호트 라벨은
# 안 건드리고, 유전자 수를 늘리되 이번엔 "같은 소스를 더" 대신 "다른 종류의 외부 지식"을
# 추가한다: JCI Insight 5개 데이터셋 전원일치(all5consistent, 1680개, top_n=1500보다 엄격한
# 기준) + Bailey 2016/Moffitt 2015 PDAC subtype 분류 문헌 유전자(815개) 합집합 = 2167개
# (--rna-genes pdac_consistency_all5_pluslit, 신규). TCGA/CPTAC Cox 회귀를 섞은
# pdac_consistency_cox_union_tcga(5시드, external 유의하게 악화 확인됨)와 달리 이건 여전히
# 100% 외부 문헌/데이터셋 기반이라 같은 실패(코호트 특이적 과적합)를 반복할 이유가 없다는
# 게 가설.
#
# 2시드(84,126)만 — 같은 목적의 pdac_consistency_2000(sbatch/porpoise_final_pdaccons2000_
# 2seed_kfold_array_hpc.sh)과 함께 방향성만 빠르게 확인.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PORPOISE_uni2native_PDACCONSALL5PL_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PORPOISE_uni2native_PDACCONSALL5PL_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/porpoise_final_pdacconsall5pluslit_2seed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2native_PDACCONSALL5PL_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1_kfold5_fold${FOLD}.log"

echo "=== PORPOISE final recipe(pdac_consistency_all5_pluslit) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes pdac_consistency_all5_pluslit --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906porpoise_final_pdacconsall5pluslit_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== PORPOISE final recipe(pdac_consistency_all5_pluslit) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
