#!/bin/bash
#SBATCH --job-name=PVT-M4PMA-NONYSTROM-1run
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/m4_pma_no_nystrom_kfold.log

# M4/PMA ablation 3종 중 Nystrom self-attention 담당(m4_pma_no_coattn_kfold_hpc.sh 상단 주석
# 참조 — Nystrom/ABMIL/co-attention 중 무엇이 문제인지 분리). m4_pma_3seed_kfold_array_hpc.sh
# (최종 채택 레시피)에서 --skip-patch-vit만 추가 — patch 토큰이 self-attention 컨텍스트 블록
# (NystromAttention, models/vit_encoder.py)을 아예 안 거치고 CNN 원본 특징 그대로 pooling
# 단계(MultiComponentPooling/co-attention)로 들어간다. 나머지(backbone/clinical/rna-aux/
# dispersion/combine-mode)는 전부 동일.
#
# 2026-08-31: 단일 seed x 단일 fold(사용자 지시 — 정식 검정 전 방향성만 빠르게 확인).
#
# 태그: PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT (model_prefix 부착 순서상
# --skip-patch-vit는 --drop-component보다도 뒤에 붙음 — train.py 확인함).
#
# 제출: sbatch sbatch/final_m1m4_3seed_kfold_hpc/m4_pma_no_nystrom_kfold_hpc.sh
# (seed/fold 바꾸려면 아래 SEED=/FOLD= 두 줄만 수정 — m4_pma_no_coattn/no_abmil과 같은 값으로
# 맞춰야 3종 비교가 성립함)

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=84
FOLD=0
N_FOLDS=5

log="paper/.hpc/train_tcga_seed${SEED}_PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT_kfold5_fold${FOLD}.log"

echo "=== M4/PMA-NONYSTROM(self-attn 컨텍스트 블록 없음, CNN 원본 특징 그대로) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --dataset tcga --external --seed "${SEED}" \
    --rna-genes literature_1500_intersection \
    --backbone uni2native \
    --clinical-staging --clinical-margin \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --combine-mode cox_add \
    --skip-patch-vit \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0831_m4pma_no_nystrom_1run 2>&1 | tee "${log}"
echo "=== M4/PMA-NONYSTROM seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
