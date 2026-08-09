#!/bin/bash
#SBATCH --job-name=PVT-PMA-uni2-coxadd-stg-ext-eval
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_uni2_coxadd_stg_multiseed_external_eval.log

# pma_uni2_coxadd_stg_multiseed_kfold_array_hpc.sh(3seed x 5fold=15개 학습)가 이미 저장해 둔
# checkpoint 15개를 재학습 없이 다시 불러와, --eval-external-ckpt(train.py, 2026-08-08 추가)로
# external(cptac) 예측만 다시 뽑아 .logs/external_preds/에 CSV로 저장한다.
#
# [왜 재학습이 아니라 재평가만 해도 되는가] external 코호트(cptac)는 어떤 seed/fold 조합으로
# 학습해도 학습 데이터에 전혀 안 들어간다 — tcga만 5-fold로 나누므로 cptac은 매번 완전히
# 배제된 채로 남는다. 그래서 이미 저장된 checkpoint의 가중치만 다시 불러와 external 전체(144명)에
# 대해 forward만 한 번 더 돌리면 되고, 학습 루프(수 시간)를 다시 돌 필요가 없다 — 15개를
# 순차로 다 돌아도 GPU 몇 분이면 끝난다(--time=1h은 넉넉한 안전 마진).
#
# checkpoint 파일명은 train.py의 tag 생성 로직(seed/rna-genes/SS/AUX/STG/R/model_prefix 전부
# 반영)을 그대로 다시 계산하는 대신, seed와 FOLD{f}OF{n} 부분만 고정하고 나머지는 와일드카드로
# 찾는다 — 이 레시피(--PMA --backbone uni2 --combine-mode cox_add 등)로 저장된 checkpoint가
# seed+fold 조합당 정확히 1개만 있다는 전제(다른 레시피를 같은 seed/fold로 따로 학습해 checkpoint
# 디렉터리에 여러 개가 쌓여 있으면 매칭이 꼬일 수 있음 — 그런 경우 아래 glob을 더 구체적으로
# 좁혀야 한다).
#
# 완료 후(.logs/external_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_uni2_coxadd_stg_multiseed_external_eval_hpc.sh
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
    # 2026-08-08: 원래 글롭(*FOLD..OF..*best_pma.pt)이 너무 느슨해 --PMA --no-clinical(M3,
    # sbatch/m3_uni2_multiseed_kfold_array_hpc.sh)도 checkpoint 접미사가 똑같이 best_pma.pt라
    # 같은 seed/fold 조합이면 잘못 매칭되는 사고가 있었다(RuntimeError: clinical 관련 buffer
    # missing — M3 checkpoint엔 age_mean/margin_mean 등이 아예 없음). 이 레시피의 정확한 태그
    # (STG_R_DISP_COX_ADD)까지 FOLD 앞에 고정해 M3/다른 PMA 변형과 절대 안 섞이게 좁힌다.
    mapfile -t MATCHES < <(ls models/checkpoint/survival_tcga_uni2_seed${SEED}_*STG_R_DISP_COX_ADD_FOLD${FOLD}OF${N_FOLDS}_best_pma.pt 2>/dev/null)
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
        --fold "${FOLD}" --n-folds "${N_FOLDS}" \
        --eval-external-ckpt "${CKPT}"
    echo "=== external eval-only: seed=${SEED} fold=${FOLD} Complete: $(date) ==="
  done
done
