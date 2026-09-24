#!/bin/bash
#SBATCH --job-name=PVT-M4SA-clusterpool-extraseeds
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_clusterpool_pdaccons_selfattnfusion_m4_extraseeds_kfold_array_%a.log

# 2026-09-24(2차): pma_clusterpool_pdaccons_final_recipe_extraseeds_kfold_array_hpc.sh(M4)를
# 5시드로 늘리면서 M4-SA(pma_clusterpool_pdaccons_selfattnfusion_m4_2seed_kfold_array_hpc.sh)를
# 2시드로 남겨두면, "M4-SA -> M4 paired bootstrap"이 서로 다른 시드 수 위에서 이뤄지는 셈이라
# M4 쪽만 분산이 줄어든 상태로 비교하게 된다 — Section 4.5(단일/소수 시드로는 신뢰 못 함)를
# 스스로 어기는 것은 M4/M7뿐 아니라 이 ablation 비교 자체에도 적용된다(사용자 지적). M4와
# 동일하게 시드 3개(42, 168, 210)를 추가해 84/126과 합쳐 총 5시드로 맞춘다. 레시피는
# pma_clusterpool_pdaccons_selfattnfusion_m4_2seed_kfold_array_hpc.sh와 완전히 동일 — SEEDS
# 배열과 --array 범위만 다르다.
#
# 3seed(42,168,210) x 5fold = 15 array task.
#
# 완료 후(기존 84/126과 합쳐 5seed로 풀링, M4의 5seed 풀링과 나란히 비교):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_SELFATTNFUSION_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,42,168,210 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_SELFATTNFUSION_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,42,168,210 --n-folds 5 --bootstrap 2000
# 5seed 기준 M4-SA -> M4 paired bootstrap(둘 다 --seeds 84,126,42,168,210):
#   python scripts/paired_bootstrap_delta.py --split external --dataset cptac \
#       --model-a PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_SELFATTNFUSION_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --model-b PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,42,168,210 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_clusterpool_pdaccons_selfattnfusion_m4_extraseeds_kfold_array_hpc.sh

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

log=".logs/train_tcga_m4sa_pma_clusterpool_selfattnfusion_extraseeds_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M4-SA(PMA ClusterPool, self-attn-fusion, extra seed) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --cluster-pool --self-attn-fusion --combine-mode cox_add --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0924pma_clusterpool_pdaccons_selfattnfusion_m4_extraseeds_kfold5_array 2>&1 | tee "${log}"
echo "=== M4-SA(PMA ClusterPool, self-attn-fusion, extra seed) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
