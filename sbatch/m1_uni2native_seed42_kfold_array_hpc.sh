#!/bin/bash
#SBATCH --job-name=PVT-M1-uni2native-seed42-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m1_uni2native_seed42_kfold_array_%a.log

# m1_uni2native_seed42_fold0_pilot_hpc.sh(array index 0, aux-free)의 fold0/seed42 파일럿에서
# val_c_index가 기존 uni2(0.3820) 대비 뚜렷이 올라 5-fold 전체로 확장한다. aux 없이
# 순수 WSI-only 상태(모달리티 사다리 순수성 유지)에서 backbone만 uni2(우리 파이프라인,
# 512 리사이즈)->uni2native(256px@0.5MPP, DX-only/coords-스케일 confound 없는 재타일링)로
# 바꾼 것 외에는 m1_novit_seed42_kfold_array_hpc.sh와 동일한 레시피.
#
# 완료 후(5개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 5개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M1_uni2native_DISP_NOVIT --seeds 42 --n-folds 5 --bootstrap 2000
#
# 비교 기준(기존 uni2, aux-free): internal pooled 0.4541 (fold0 단독 test_c_index=0.4137)
#
# 제출: sbatch sbatch/m1_uni2native_seed42_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=42
N_FOLDS=5
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_M1_uni2native_DISP_NOVIT_kfold5_fold${FOLD}.log"

echo "=== M1(uni2native,skip-patch-vit,DISP, aux-free) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M1 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --attn-dispersion \
    --skip-patch-vit \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m1_uni2native_seed42_kfold5_array 2>&1 | tee "${log}"
echo "=== M1(uni2native,skip-patch-vit,DISP, aux-free) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
