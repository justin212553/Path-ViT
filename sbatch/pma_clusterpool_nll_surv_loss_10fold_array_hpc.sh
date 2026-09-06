#!/bin/bash
#SBATCH --job-name=PVT-PMA-clusterpool-nllsurv-10fold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_clusterpool_nll_surv_loss_10fold_array_%a.log

# 2026-09-06: sbatch/pma_nll_surv_loss_10fold_array_hpc.sh와 완전히 동일하지만 --cluster-pool만
# 추가 — models/vit_pma.py::ViT_PMA cluster_pool=True(2026-09-05 도입, Nystrom+ABMIL을 사전계산
# unsupervised 군집 대표값(K=10)으로 대체). 별도 모델 클래스가 아니라 --PMA의 플래그 하나라
# 나머지 레시피(rna-genes/combine-mode/CLR100/CNV/mutation/nll_surv 등)는 그대로 유지.
#
# --cluster-centroids-path 기본값(None -> data/cluster_centroids_{backbone}.pt)을 그대로
# 쓴다 — backbone=uni2native면 data/cluster_centroids_uni2native_k10_tcgacptac.pt(로컬 확인
# 완료, CPU 생성 테스트로 실제 로딩까지 검증됨).
#
# 완료 후: 정확한 모델 태그는 `ls .logs/kfold_preds/tcga_PMA*CLUSTERPOOL*NLLSURV4*`로 확인 후
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model <확인한 태그> --seeds 84 --n-folds 10 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_clusterpool_nll_surv_loss_10fold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=84
N_FOLDS=10
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_PMA_uni2native_INT1500_CNV_SS_AUX_STG_R_MUT_CLUSTERPOOL_NLLSURV4_kfold10_fold${FOLD}.log"

echo "=== PMA+ClusterPool nll_surv loss(uni2native,cox_add,CLR100,CNV,MUT) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --cluster-pool \
    --surv-loss nll_surv --nll-n-bins 4 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906pma_clusterpool_nllsurv_10fold_array 2>&1 | tee "${log}"
echo "=== PMA+ClusterPool nll_surv loss(uni2native,cox_add,CLR100,CNV,MUT) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
