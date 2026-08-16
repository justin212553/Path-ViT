#!/bin/bash
#SBATCH --job-name=PVT-M4-clrlr20-rlr20-seed42-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m4_clrmult20_rlrmult20_seed42_kfold_array_%a.log

# 2026-08-15: M4-NOVIT(WSI+Clinical+RNA, cox_add)에 --clinical-lr-mult 20.0 + --rna-lr-mult 20.0
# 동시 적용 — M2/M3 각각에서 확인된 개선(clinical/RNA 브랜치가 WSI와 같은 lr로 경쟁하면
# 밀려난다는 가설)을 M4에 합쳐서 적용. 애초 문제의식(M4가 M3를 못 넘는 게 clinical 결합의
# 문제)을 정면으로 다루는 실험. fold1 파일럿(로컬): internal 0.6348->0.6631,
# external 0.5799->0.6380 확인 — M3 baseline external(0.5841)도 이 fold에서는 넘었다.
#
# 완료 후(5개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 5개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT_CLR20_RLR20 --seeds 42 --n-folds 5 --bootstrap 2000
#
# 비교 기준(lr-mult 없는 M4-NOVIT baseline, seed42 fold1 단독): internal=0.6348, external=0.5799
# (참고, M3 baseline seed42 5-fold pooled: internal=0.6488, external=0.5667)
#
# 제출: sbatch sbatch/m4_clrmult20_rlrmult20_seed42_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=42
N_FOLDS=5
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT_CLR20_RLR20_kfold5_fold${FOLD}.log"

echo "=== M4(uni2,cox_add,STG+R,DISP,skip-patch-vit,clr20+rlr20) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4 --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --skip-patch-vit \
    --clinical-lr-mult 20.0 --rna-lr-mult 20.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m4_clrmult20_rlrmult20_seed42_kfold5_array 2>&1 | tee "${log}"
echo "=== M4(uni2,cox_add,STG+R,DISP,skip-patch-vit,clr20+rlr20) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
