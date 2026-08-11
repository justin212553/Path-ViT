#!/bin/bash
#SBATCH --job-name=PVT-M4A-uni2-coxadd-stg-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m4a_uni2_coxadd_stg_r_multiseed_kfold_array_%a.log

# ViT_M4A(models/vit_m4a.py) — RNA가 query, pooling 이전 raw patch 토큰이 key/value인
# co-attention(Chen et al. 2021 MCAT 스타일). PMA(다성분 pooling 이후 co-attention)와 달리
# top-k/mean/std 같은 손수 짠 통계 없이 patch 전체 위에서 바로 RNA-guided pooling을 한다 —
# "top-k가 노이즈를 증폭시키는 게 아닐까"라는 가설에서 나온 대안 구조.
#
# 2026-08-11: margin(R)/staging(STG)/combine_mode(cox_add)/attn_dispersion을 ViT_M4/ViT_M4A에
# 새로 이식해서(models/vit_m4.py), baseline PMA와 완전히 동일한 최종 레시피로 처음 테스트한다 —
# findings_backlog.md의 예전 M4A 기록들은 이 레시피 이전 것이라 직접 비교가 안 됐었다.
#
# seed=84 단일 split 로컬 파일럿 결과(2026-08-11): internal 0.5772(PMA 0.6484 대비 -0.0712),
# external 0.6261(PMA 0.5993 대비 +0.0268) — external은 올랐지만 internal이 크게 떨어지는
# 트레이드오프, 게다가 지금 목표(internal을 M6=0.6586 수준까지)에 안 맞는 방향. NOTOP 등
# 단일 파일럿 신호가 멀티시드에서 재현 안 된 전례가 많아 큰 기대 없이 확인 차원으로 돌린다.
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5 (baseline과 동일한 15-job 관례).
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M4A_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m4a_uni2_coxadd_stg_r_multiseed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(42 84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_M4A_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== M4A(uni2,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4A --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0811m4a_uni2_coxadd_stg_r_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== M4A(uni2,cox_add,STG+R) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
