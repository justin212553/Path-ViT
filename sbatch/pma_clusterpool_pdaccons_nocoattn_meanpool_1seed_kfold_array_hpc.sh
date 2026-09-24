#!/bin/bash
#SBATCH --job-name=PVT-M4-clusterpool-meanpool
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_clusterpool_pdaccons_nocoattn_kfold_array_%a.log

# 2026-09-24: 논문 리뷰 피드백(Q1 후반) — "클러스터링 자체의 가치"를 방어하는 세 번째 fusion
# ablation. M4-SA(self-attn-fusion, K개 cluster token이 서로를 참조)와 다르게, 이건 K개
# cluster token을 아예 서로 참조도 안 시키고 단순 평균(mean-pool)만 한다 — 기존에 이미 존재하던
# train.py --no-coattn(models/vit_pma.py ViT_PMA use_coattn=False)이 cluster_pool=True에서도
# 그대로 이 의미로 동작한다(components.mean(dim=0)) — 신규 코드 불필요.
#
# 2026-09-24(2차, 시간 절약 — 사용자 지시): 이것도 K 민감도(pma_clusterpool_pdaccons_k5/k20_
# 1seed_...)와 같은 방향성 확인용 ablation이라 M4-SA/시드확장(A/C)과 달리 paired significance가
# 목적이 아니다 — 2seed가 아니라 seed=84 1개만 돌린다.
#
# sbatch/pma_clusterpool_pdaccons_final_recipe_2seed_kfold_array_hpc.sh(M4)에서 --no-coattn
# 하나만 추가하고 seed를 84 하나로 줄인 것 — 나머지 레시피는 완전히 동일.
#
# 1seed(84) x 5fold = 5 array task.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_NOCOATTN_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_NOCOATTN_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84 --n-folds 5 --bootstrap 2000
# (정확한 태그는 실제 생성된 CSV로 확인 권장:
#  `ls .logs/kfold_preds/tcga_PMA_uni2native_PDACCONS1500*NOCOATTN*CLUSTERPOOL*`)
#
# 제출: sbatch sbatch/pma_clusterpool_pdaccons_nocoattn_meanpool_1seed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_m4_clusterpool_nocoattn_meanpool_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M4(PMA ClusterPool, no-coattn/mean-pool, 1seed) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --cluster-pool --no-coattn --combine-mode cox_add --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0924pma_clusterpool_pdaccons_nocoattn_meanpool_1seed_kfold5_array 2>&1 | tee "${log}"
echo "=== M4(PMA ClusterPool, no-coattn/mean-pool, 1seed) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
