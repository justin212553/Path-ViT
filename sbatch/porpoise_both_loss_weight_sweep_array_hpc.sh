#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-bothloss-wsweep
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-39
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_both_loss_wsweep_array_%a.log

# 2026-09-06: sbatch/porpoise_both_loss_10fold_array_hpc.sh(--nll-cox-weight 1.0, 동등가중)의
# paired bootstrap 결과 — nll_surv 단독 대비 external delta=+0.0091, p=0.554로 비유의. 가중치
# 1.0이 최적이라는 근거가 전혀 없었으므로(그냥 기본값을 그대로 썼을 뿐), 다른 지점에서 더
# 나은(또는 통계적으로 유의한) 조합이 있는지 스윕한다.
#
# --nll-cox-weight in {0.1, 0.3, 3, 10} — 1.0은 이미 있어서 제외. 0.1/0.3은 "nll_surv가 주,
# cox가 보조 정규화" 쪽, 3/10은 "cox가 주, nll_surv가 보조" 쪽으로 양방향을 다 본다(0에
# 가까울수록 nll_surv 단독, 크게 갈수록 cox 단독에 근접 — 단, nll_surv 항은 가중치 무관하게
# 항상 더해지므로 진짜 "cox 단독"과는 다름, sbatch/porpoise_cox_loss_matched_10fold_array_hpc.sh
# 가 그 진짜 cox-단독 비교 대상). 정수 가중치는 일부러 "3.0" 대신 "3"으로 써서 train.py의
# model_prefix 접미사 포맷(f"{weight:g}" — 3.0을 "3"으로 씀)과 그대로 맞아떨어지게 했다.
#
# 4 weight x 2 seed(84,126) x 5 fold = 40 array task.
#
# 완료 후, 각 가중치의 태그로 pool + paired bootstrap(nll_surv 단독/cox 단독/weight=1.0과 각각
# 비교):
#   for W in 0.1 0.3 3 10; do
#     python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#         --model PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX${W} \
#         --seeds 84,126 --n-folds 5 --bootstrap 2000
#     python scripts/pool_multiseed_external_preds.py --dataset cptac \
#         --model PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX${W} \
#         --seeds 84,126 --n-folds 5 --bootstrap 2000
#   done
#
# 제출: sbatch sbatch/porpoise_both_loss_weight_sweep_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

WEIGHTS=(0.1 0.3 3 10)
SEEDS=(84 126)
N_FOLDS=5
N_SEEDS=${#SEEDS[@]}
PER_WEIGHT=$((N_SEEDS * N_FOLDS))

IDX=$SLURM_ARRAY_TASK_ID
WEIGHT_IDX=$((IDX / PER_WEIGHT))
REM=$((IDX % PER_WEIGHT))
SEED_IDX=$((REM / N_FOLDS))
FOLD=$((REM % N_FOLDS))
WEIGHT=${WEIGHTS[$WEIGHT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX${WEIGHT}_kfold5_fold${FOLD}.log"

echo "=== PORPOISE both loss(nll_cox_weight=${WEIGHT}) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight "${WEIGHT}" \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906porpoise_bothloss_wsweep_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== PORPOISE both loss(nll_cox_weight=${WEIGHT}) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
