#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-bothloss-extraseeds-int1500
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_both_loss_extraseeds_int1500_kfold_array_%a.log

# 2026-09-06: sbatch/porpoise_both_loss_10fold_array_hpc.sh(확정 최종 레시피, literature_1500,
# seed 84/126)와 완전히 동일한 레시피를 seed 168/210/252로 3개 더 돌린다. 목적은 RNA 세트
# 비교(literature_1500 vs pdac_consistency_1500)의 internal 델타(-0.034, p=0.23, 비유의)가
# 2시드(n=2)만으로 판단하기엔 너무 좁은 표본이었기 때문 — internal 앙상블은 환자당 seed 수만큼만
# 평균되므로(fold와 달리 external처럼 10-way가 아니라 seed 수만큼의 n-way), 시드를 늘리면
# (1) "seed 변동폭" 추정 자체의 신뢰도가 오르고 (2) internal 앙상블 노이즈도 직접 줄어든다.
#
# seed=42는 제외 — WSI-포함 모델에서 비정상적으로 분산이 크다고 이미 확인된 이상치 시드라
# 이번처럼 노이즈를 줄이려는 목적에 다시 섞으면 역효과(사용자 결정, 2026-09-06).
#
# pdac_consistency_1500 쪽 동일 추가 시드는 sbatch/porpoise_final_pdacconsistency_extraseeds_kfold_array_hpc.sh.
#
# 완료 후, 기존 84/126 결과와 합쳐 5seed 기준으로 재pooling:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,168,210,252 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,168,210,252 --n-folds 5 --bootstrap 2000
# pdac_consistency_1500 쪽(동일하게 5seed로 pooling한 뒤)과 paired bootstrap 재비교:
#   python scripts/paired_bootstrap_delta.py --split internal --dataset tcga \
#       --model-a PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --model-b PORPOISE_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,168,210,252 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/porpoise_both_loss_extraseeds_int1500_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(168 210 252)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1_kfold5_fold${FOLD}.log"

echo "=== PORPOISE both loss(literature_1500) extra-seed seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906porpoise_bothloss_int1500_extraseeds_kfold5_array 2>&1 | tee "${log}"
echo "=== PORPOISE both loss(literature_1500) extra-seed seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
