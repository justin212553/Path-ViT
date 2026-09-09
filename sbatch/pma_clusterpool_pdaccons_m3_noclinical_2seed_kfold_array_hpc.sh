#!/bin/bash
#SBATCH --job-name=PVT-M3-clusterpool
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_clusterpool_pdaccons_m3_noclinical_kfold_array_%a.log

# 2026-09-09: 논문 M1~M7 ablation 테이블 최종 레시피 재작성 중 M3(WSI+RNA, Clinical 제외) 슬롯.
# --PMA --cluster-pool --no-clinical — M4(=pma_clusterpool_pdaccons_final_recipe, sbatch/
# pma_clusterpool_pdaccons_final_recipe_2seed_kfold_array_hpc.sh)에서 clinical만 빼고 나머지
# (pdac_consistency_1500, CNV, both-loss)는 그대로. clinical이 없으므로 mutation/staging/margin/
# CLR100도 해당 없음(mutation은 use_clinical=True의 cox_add 가산항에만 배선돼 있음, models/
# vit_pma.py). combine_mode는 기본값 concat 그대로 둔다(no-clinical이면 cox_add는 애초에
# ValueError로 막혀 있음).
#
# 2seed(84,126) x 5fold.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PMA_uni2native_PDACCONS1500_CNV_CLUSTERPOOL_NOCLINICAL_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PMA_uni2native_PDACCONS1500_CNV_CLUSTERPOOL_NOCLINICAL_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
# (정확한 태그는 실제 생성된 CSV로 확인 권장:
#  `ls .logs/kfold_preds/tcga_PMA_uni2native_PDACCONS1500*CLUSTERPOOL*NOCLINICAL*`)
#
# 제출: sbatch sbatch/pma_clusterpool_pdaccons_m3_noclinical_2seed_kfold_array_hpc.sh

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

log=".logs/train_tcga_m3_pma_clusterpool_noclinical_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M3(PMA ClusterPool, no-clinical) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --cluster-pool --no-clinical --rna-genes pdac_consistency_1500 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --use-cnv \
    --patch-keep-frac 0.8 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0909m3_pma_clusterpool_noclinical_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== M3(PMA ClusterPool, no-clinical) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
