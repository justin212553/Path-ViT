#!/bin/bash
#SBATCH --job-name=PVT-M4SA-clusterpool
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_clusterpool_pdaccons_selfattnfusion_m4_kfold_array_%a.log

# 2026-09-24: 논문 리뷰 피드백(Q1) — M2->M4에서 (a) RNA-seq 추가와 (b) self-attention->
# co-attention 전환이 동시에 일어나 "RNA-guided co-attention이 internal-external gap을
# 좁힌다"는 핵심 주장이 fusion 메커니즘 때문인지 단순히 RNA 정보가 들어와서인지 분리가 안
# 된다는 지적에 대한 직접 검증. M4-SA — sbatch/pma_clusterpool_pdaccons_final_recipe_
# 2seed_kfold_array_hpc.sh(M4, 확정 레시피)에서 --self-attn-fusion 하나만 추가한 것.
# z_rna는 M4와 동일하게 risk_head에 직결 concat되지만(train.py --self-attn-fusion, models/
# vit_pma.py ViT_PMA.self_attn_fusion, 2026-09-24 신규 구현), K=11 cluster token을 모으는
# 방식이 RNA-query co-attention이 아니라 M1/M2와 동일한 query-free self-attention pooling
# (models/self_attention_pooling.py)이다. M4-SA -> M4 paired bootstrap이 유의하게 다르면
# co-attention(fusion 메커니즘) 자체의 기여, 다르지 않으면 RNA가 risk_head에 concat된 것만으로
# 충분하다는 뜻.
#
# 나머지 레시피는 M4와 완전히 동일(pdac_consistency_1500, CNV, mutation, staging+margin,
# CLR100, both-loss, patch-keep-frac 0.8) — self-attn-fusion 하나만 다르다.
#
# 2seed(84,126) x 5fold — M4와 동일 시드/fold라 paired bootstrap이 성립한다.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_SELFATTNFUSION_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_SELFATTNFUSION_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 train.py의 여러 조건부 접미사 순서를 직접 뽑은 추정치 — 실제 생성된 CSV로
#  확인 권장: `ls .logs/kfold_preds/tcga_PMA_uni2native_PDACCONS1500*SELFATTNFUSION*`)
# 논문 핵심 주장을 직접 검증하는 paired bootstrap(M4-SA -> M4):
#   python scripts/paired_bootstrap_delta.py --split external --dataset cptac \
#       --model-a PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_SELFATTNFUSION_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --model-b PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_clusterpool_pdaccons_selfattnfusion_m4_2seed_kfold_array_hpc.sh

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

log=".logs/train_tcga_m4sa_pma_clusterpool_selfattnfusion_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M4-SA(PMA ClusterPool, self-attn-fusion) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --cluster-pool --self-attn-fusion --combine-mode cox_add --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0924pma_clusterpool_pdaccons_selfattnfusion_m4_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== M4-SA(PMA ClusterPool, self-attn-fusion) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
