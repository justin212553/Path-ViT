#!/bin/bash
#SBATCH --job-name=PVT-PMA-clusterpool-pdaccons
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_clusterpool_pdaccons_final_recipe_kfold_array_%a.log

# 2026-09-07: BRCA에서 ClusterPool이 matched-loss 조건으로 M7을 유의하게 이긴 뒤(delta+0.061,
# p=0.009 — PORPOISE의 plain gated-ABMIL은 비유의였던 것과 대조) — 아키텍처를 하나로 정착시키기
# 위해 PAAD에서도 ClusterPool을 확정 레시피 총동원으로 시도한다(사용자 지시). 목표: pdac_
# consistency_1500에서 우리 PORPOISE 최고점(internal 0.618/external 0.619, 5seed)에 비빌 수
# 있는지.
#
# 확정 레시피(PORPOISE 최종과 동일하게 최대한 맞춤):
#   --PMA --cluster-pool, uni2native, pdac_consistency_1500, CNV, mutation, staging+margin,
#   CLR100, --surv-loss both --nll-cox-weight 1.0, patch-keep-frac 0.8
# 뺀 것 — --attn-dispersion: cluster_pool=True는 forward()가 일찍 return해서 attn-dispersion/
# spatial-autocorr 블록을 아예 안 타므로 같이 쓰면 ValueError(models/vit_pma.py 가드,
# 2026-09-05 확인된 사양) — 애초에 호환 안 됨.
# AUX(--rna-aux-weight)도 안 켠다 — PAAD 자체 최종 결론(no_aux)을 그대로 따름(BRCA의 기본
# AUX=1.0 관례는 BRCA 전용, PAAD엔 안 맞음).
#
# 2seed(84,126)로 시작 — PAAD에서 cluster_pool+CNV+mutation+both-loss 조합 자체가 처음이라
# 방향성부터 확인. 처음 도는 조합이니 array 전체보다 첫 task 로그부터 확인 권장.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 train.py의 여러 조건부 접미사 순서를 직접 뽑은 추정치 — 실제 생성된 CSV로
#  확인 권장: `ls .logs/kfold_preds/tcga_PMA_uni2native_PDACCONS1500*CLUSTERPOOL*`)
# PORPOISE 최종(pdac_consistency_1500)과 paired bootstrap 비교:
#   python scripts/paired_bootstrap_delta.py --split external --dataset cptac \
#       --model-a PORPOISE_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --model-b PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_clusterpool_pdaccons_final_recipe_2seed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1_kfold5_fold${FOLD}.log"

echo "=== PMA ClusterPool(pdac_consistency_1500, 확정 레시피) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --cluster-pool --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0907pma_clusterpool_pdaccons_final_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== PMA ClusterPool(pdac_consistency_1500, 확정 레시피) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
