#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M4SA-multitss
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_clusterpool_selfattnfusion_multitss_kfold_array_%a.log

# 2026-09-24: PAAD 쪽 pma_clusterpool_pdaccons_selfattnfusion_m4_2seed_kfold_array_hpc.sh와 짝을
# 이루는 BRCA M4-SA — sbatch/brca_m4_clusterpool_multitss_bothloss_kfold_array_hpc.sh(M4, 확정
# 레시피)에서 --self-attn-fusion 하나만 추가(scripts/train_brca_m4.py, 2026-09-24 신규 플래그).
# 논문 리뷰 피드백(Q1)의 co-attention vs RNA-존재-효과 분리 검증을 BRCA에서도 반복 — M2->M4
# pair는 이미 external에서 유의(+0.0659, p=0.028)했던 만큼, 이 비교가 그 유의성이 fusion
# 메커니즘 때문인지 확인하는 데 특히 중요하다.
#
# 2seed(84,126) x 5fold — M4와 동일 시드/fold.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_SELFATTNFUSION_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_SELFATTNFUSION_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1 --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 실제 생성된 CSV로 확인 권장: `ls .logs/kfold_preds/brca_BRCA_PMA_CONS*SELFATTNFUSION*`)
# 논문 핵심 주장을 직접 검증하는 paired bootstrap(M4-SA -> M4):
#   python scripts/paired_bootstrap_delta.py --split external --dataset brca \
#       --model-a BRCA_PMA_CONS882_STG_SELFATTNFUSION_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1 \
#       --model-b BRCA_PMA_CONS882_STG_CLUSTERPOOL_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/brca_m4_clusterpool_selfattnfusion_multitss_bothloss_kfold_array_hpc.sh

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

log=".logs/train_brca_m4sa_clusterpool_selfattnfusion_multitss_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M4-SA ClusterPool(self-attn-fusion, multi-TSS external) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --clinical-staging --cluster-pool --self-attn-fusion --clinical-lr-mult 100 \
    --external-tss multi \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --group-ts 0924_brca_m4_clusterpool_selfattnfusion_multitss_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M4-SA ClusterPool(self-attn-fusion, multi-TSS external) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
