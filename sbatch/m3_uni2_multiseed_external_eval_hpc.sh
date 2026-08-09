#!/bin/bash
#SBATCH --job-name=PVT-M3-uni2-ext-eval
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m3_uni2_multiseed_external_eval.log

# m3_uni2_multiseed_kfold_array_hpc.sh(3seed x 5fold=15개 학습)가 저장해 둔 checkpoint 15개를
# 재학습 없이 다시 불러와 --eval-external-ckpt로 external(cptac) 예측만 재추출해
# .logs/external_preds/에 CSV로 저장한다. M3는 --PMA --no-clinical이라 checkpoint 접미사는
# best_pma.pt(PMA와 동일 클래스) — glob이 PMA 본체 checkpoint와 섞이지 않도록 NOCLINICAL 태그로 좁힌다.
#
# 완료 후(.logs/external_preds/에 CSV 15개 확인):
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PMA_uni2_INT1500_SS_AUX_NOCLINICAL_DISP --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출(15개 학습 job이 전부 끝나 checkpoint가 다 저장된 뒤에): sbatch sbatch/m3_uni2_multiseed_external_eval_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(42 84 126)
N_FOLDS=5

for SEED in "${SEEDS[@]}"; do
  for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    mapfile -t MATCHES < <(ls models/checkpoint/survival_tcga_uni2_seed${SEED}_*NOCLINICAL*FOLD${FOLD}OF${N_FOLDS}*best_pma.pt 2>/dev/null)
    if [ "${#MATCHES[@]}" -eq 0 ]; then
      echo "[SKIP] seed=${SEED} fold=${FOLD}: checkpoint를 못 찾음 (학습이 아직 안 끝났거나 경로가 다름)"
      continue
    fi
    if [ "${#MATCHES[@]}" -gt 1 ]; then
      echo "[경고] seed=${SEED} fold=${FOLD}: checkpoint가 ${#MATCHES[@]}개 매칭됨 — 첫 번째만 사용: ${MATCHES[0]}"
    fi
    CKPT="${MATCHES[0]}"

    echo "=== external eval-only: seed=${SEED} fold=${FOLD} ckpt=${CKPT} Start: $(date) ==="
    python -u ./train.py --PMA --no-clinical --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" \
        --eval-external-ckpt "${CKPT}"
    echo "=== external eval-only: seed=${SEED} fold=${FOLD} Complete: $(date) ==="
  done
done
