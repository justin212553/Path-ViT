#!/bin/bash
#SBATCH --job-name=PVT-PMA-uni2-lr1e4wd1e2-ext-eval
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_uni2_coxadd_stg_lr1e4_wd1e2_multiseed_external_eval.log

# pma_uni2_coxadd_stg_lr1e4_wd1e2_multiseed_kfold_array_hpc.sh(3seed x 5fold=15개 학습)가 저장한
# checkpoint 15개를 재학습 없이 다시 불러와 --eval-external-ckpt로 external(cptac) 예측만 뽑는다.
# 원리는 baseline용 pma_uni2_coxadd_stg_multiseed_external_eval_hpc.sh와 완전히 동일 — cptac은
# 어떤 seed/fold 조합으로 학습해도 학습 데이터에 안 들어가므로 재학습 없이 forward만 하면 된다.
#
# [체크포인트 glob 주의] 2026-08-08에 이 프로젝트에서 두 번 반복된 사고(글롭이 너무 느슨해
# M3/NOTOP 등 다른 레시피의 checkpoint가 잘못 매칭됨) 재발 방지를 위해, 이 레시피의 정확한 태그
# (STG_R_DISP_COX_ADD_LR1e-04_WD1e-02, FOLD 바로 앞)까지 전부 고정해서 찾는다 — baseline
# (LR/WD 태그 없음)이나 NOTOP 변형과 절대 안 섞이게.
#
# 완료 후(.logs/external_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_LR1e-04_WD1e-02 \
#       --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_uni2_coxadd_stg_lr1e4_wd1e2_multiseed_external_eval_hpc.sh
# (15개 학습 job이 전부 끝나 checkpoint가 다 저장된 뒤에 제출할 것)

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(42 84 126)
N_FOLDS=5

for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    mapfile -t MATCHES < <(ls models/checkpoint/survival_tcga_uni2_seed${SEED}_*STG_R_DISP_COX_ADD_LR1e-04_WD1e-02_FOLD${FOLD}OF${N_FOLDS}_best_pma.pt 2>/dev/null)
    if [ "${#MATCHES[@]}" -eq 0 ]; then
      echo "[SKIP] seed=${SEED} fold=${FOLD}: checkpoint를 못 찾음 (학습이 아직 안 끝났거나 경로가 다름)"
      continue
    fi
    if [ "${#MATCHES[@]}" -gt 1 ]; then
      echo "[경고] seed=${SEED} fold=${FOLD}: checkpoint가 ${#MATCHES[@]}개 매칭됨 — 첫 번째만 사용: ${MATCHES[0]}"
    fi
    CKPT="${MATCHES[0]}"

    echo "=== external eval-only: seed=${SEED} fold=${FOLD} ckpt=${CKPT} Start: $(date) ==="
    python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --clinical-margin --clinical-staging --combine-mode cox_add \
        --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
        --lr 1e-4 --weight-decay 1e-2 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" \
        --eval-external-ckpt "${CKPT}"
    echo "=== external eval-only: seed=${SEED} fold=${FOLD} Complete: $(date) ==="
  done
done
