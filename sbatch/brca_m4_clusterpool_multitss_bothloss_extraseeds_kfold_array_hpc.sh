#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M4-multitss-extraseeds
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_clusterpool_multitss_extraseeds_kfold_array_%a.log

# 2026-09-24: PAAD 쪽 pma_clusterpool_pdaccons_final_recipe_extraseeds_kfold_array_hpc.sh(M4)와
# 짝을 이루는 BRCA 5-seed 확장 — 헤더 설명은 그 파일 참조. 레시피는
# sbatch/brca_m4_clusterpool_multitss_bothloss_kfold_array_hpc.sh(M4)와 완전히 동일 — SEEDS
# 배열과 --array 범위만 다르다.
#
# 3seed(42,168,210) x 5fold = 15 array task.
#
# 완료 후(기존 84/126과 합쳐 5seed로 풀링):
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1 --seeds 84,126,42,168,210 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1 --seeds 84,126,42,168,210 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/brca_m4_clusterpool_multitss_bothloss_extraseeds_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(42 168 210)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_brca_m4_clusterpool_multitss_extraseeds_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M4 ClusterPool(multi-TSS external, extra seed) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --clinical-staging --cluster-pool --clinical-lr-mult 100 \
    --external-tss multi \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --group-ts 0924_brca_m4_clusterpool_multitss_extraseeds_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M4 ClusterPool(multi-TSS external, extra seed) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
