#!/bin/bash
#SBATCH --job-name=PVT-M3SA-clusterpool
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_clusterpool_pdaccons_selfattnfusion_m3_kfold_array_%a.log

# 2026-09-24: pma_clusterpool_pdaccons_selfattnfusion_m4_2seed_kfold_array_hpc.sh(M4-SA)와 짝을
# 이루는 M3-SA — M1->M3 pair(WSI-only self-attention -> WSI+RNA co-attention)에 대해서도 같은
# 분리 검증을 한다. sbatch/pma_clusterpool_pdaccons_m3_noclinical_2seed_kfold_array_hpc.sh(M3,
# WSI+RNA, clinical 제외)에서 --self-attn-fusion 하나만 추가.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2native_PDACCONS1500_CNV_SELFATTNFUSION_CLUSTERPOOL_NOCLINICAL_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PMA_uni2native_PDACCONS1500_CNV_SELFATTNFUSION_CLUSTERPOOL_NOCLINICAL_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 실제 생성된 CSV로 확인 권장:
#  `ls .logs/kfold_preds/tcga_PMA_uni2native_PDACCONS1500*SELFATTNFUSION*NOCLINICAL*`)
# M3-SA -> M3 paired bootstrap(M1 자체도 함께 비교하면 co-attention 효과가 M3/M4 양쪽에서
# 일관되는지 볼 수 있음):
#   python scripts/paired_bootstrap_delta.py --split external --dataset cptac \
#       --model-a PMA_uni2native_PDACCONS1500_CNV_SELFATTNFUSION_CLUSTERPOOL_NOCLINICAL_NLLSURV4_NLLCOX1 \
#       --model-b PMA_uni2native_PDACCONS1500_CNV_CLUSTERPOOL_NOCLINICAL_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_clusterpool_pdaccons_selfattnfusion_m3_noclinical_2seed_kfold_array_hpc.sh

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

log=".logs/train_tcga_m3sa_pma_clusterpool_selfattnfusion_noclinical_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M3-SA(PMA ClusterPool, self-attn-fusion, no-clinical) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --cluster-pool --self-attn-fusion --no-clinical --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --use-cnv \
    --patch-keep-frac 0.8 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0924m3_pma_clusterpool_selfattnfusion_noclinical_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== M3-SA(PMA ClusterPool, self-attn-fusion, no-clinical) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
