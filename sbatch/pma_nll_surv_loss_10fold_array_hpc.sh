#!/bin/bash
#SBATCH --job-name=PVT-PMA-nllsurv-10fold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_nll_surv_loss_10fold_array_%a.log

# 2026-09-06: sbatch/porpoise_nll_surv_loss_10fold_array_hpc.sh와 동일한 실험을 PMA
# 아키텍처(models/vit_pma.py::ViT_PMA, co-attention+Nystrom, combine_mode="cox_add")에도
# 적용 — 레퍼런스 recipe(sbatch/pma_extraseed_kfold_array_hpc.sh, "가장 잘 나온" PMA 계열
# 기준선)에서 딱 아래만 바꿨다:
#   1. --backbone uni2 -> uni2native (PORPOISE 쪽과 동일 이유 — uni2는 구형 1024px@1.0MPP,
#      uni2native가 이 세션 전체의 정식 256px@0.5MPP 파이프라인)
#   2. --surv-loss cox(default) -> nll_surv(models/vit_pma.py 2026-09-06 신규 지원,
#      risk_head가 --nll-n-bins개 시간-구간별 hazard logit을 뱉음)
#   3. --clinical-lr-mult 100, --use-cnv, --clinical-mutation 추가(사용자 지시) —
#      mutation은 ViT_PMA에 없던 기능이라 이번에 새로 이식(combine_mode="cox_add" 전용,
#      models/vit_m4.py::ViT_M4와 동일 관례).
#
# **주의**: --rna-genes literature_1500_intersection은 PAAD에서 leaky한 유전자셋
# (findings_backlog.md) — "RNA는 그대로 두라"는 사용자 지시로 안 바꿨다. cox/nll_surv 두
# 버전 다 동일하게 부풀려져 있으므로 "loss 함수 차이"라는 이 실험의 관심사(상대 비교)는 유효.
#
# 3seed x 5fold 대신 1seed(84) x 10fold(10개 array)로 간소화 — "그냥 10폴드 한번에" 지시.
#
# 완료 후: 정확한 모델 태그는 `ls .logs/kfold_preds/tcga_PMA*NLLSURV4*`로 확인 후
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model <확인한 태그> --seeds 84 --n-folds 10 --bootstrap 2000
#
# 제출: sbatch sbatch/pma_nll_surv_loss_10fold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=84
N_FOLDS=10
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_PMA_uni2native_INT1500_CNV_SS_AUX_STG_R_MUT_NLLSURV4_kfold10_fold${FOLD}.log"

echo "=== PMA nll_surv loss(uni2native,cox_add,CLR100,CNV,MUT) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --surv-loss nll_surv --nll-n-bins 4 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906pma_nllsurv_10fold_array 2>&1 | tee "${log}"
echo "=== PMA nll_surv loss(uni2native,cox_add,CLR100,CNV,MUT) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
