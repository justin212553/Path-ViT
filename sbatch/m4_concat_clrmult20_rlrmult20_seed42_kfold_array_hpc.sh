#!/bin/bash
#SBATCH --job-name=PVT-M4-concat-clrlr20-rlr20-seed42-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m4_concat_clrmult20_rlrmult20_seed42_kfold_array_%a.log

# 2026-08-15: M4-NOVIT의 clinical 결합을 cox_add 대신 concat으로 바꾼 채(--combine-mode
# 생략 -> ViT_M4 기본값 concat) --clinical-lr-mult 20.0 + --rna-lr-mult 20.0을 같이 적용한
# 버전 — m4_clrmult20_rlrmult20_seed42_kfold_array_hpc.sh(cox_add 버전)와 짝을 이뤄 어느
# combine_mode가 더 나은지 비교한다. clinical-lr-mult는 concat에서 훨씬 크게 작동한다는 게
# M2에서 확인됐지만(cox_add는 거의 무력, concat은 internal +0.09), M4 fold1 파일럿(로컬)
# 에서는 cox_add+lr-mult(internal=0.6631/external=0.6380)와 concat+lr-mult
# (internal=0.6738/external=0.6266)가 팽팽해서 5-fold로 갈라야 한다.
#
# 완료 후(5개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 5개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M4_uni2_INT1500_SS_AUX_STG_R_DISP_NOVIT_CLR20_RLR20 --seeds 42 --n-folds 5 --bootstrap 2000
#
# 비교 기준(fold1 단독):
#   baseline(cox_add, lr-mult 없음): internal=0.6348, external=0.5799
#   cox_add+clr20+rlr20:             internal=0.6631, external=0.6380
#   concat+clr20+rlr20(이 스크립트):  internal=0.6738, external=0.6266
#
# 제출: sbatch sbatch/m4_concat_clrmult20_rlrmult20_seed42_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=42
N_FOLDS=5
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_M4_uni2_INT1500_SS_AUX_STG_R_DISP_NOVIT_CLR20_RLR20_kfold5_fold${FOLD}.log"

echo "=== M4(uni2,concat,STG+R,DISP,skip-patch-vit,clr20+rlr20) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4 --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --skip-patch-vit \
    --clinical-lr-mult 20.0 --rna-lr-mult 20.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m4_concat_clrmult20_rlrmult20_seed42_kfold5_array 2>&1 | tee "${log}"
echo "=== M4(uni2,concat,STG+R,DISP,skip-patch-vit,clr20+rlr20) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
