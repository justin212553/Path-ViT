#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-pdaccoxunion
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-24
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_final_pdaccoxunion_5seed_kfold_array_%a.log

# 2026-09-06: sbatch/porpoise_both_loss_10fold_array_hpc.sh(확정 최종 레시피 — uni2native,
# CLR100, CNV, mutation, STG+R, DISP, surv-loss both weight=1.0)에서 RNA 유전자셋만
# --rna-genes pdac_consistency_cox_union_tcga(신규, 2026-09-06)로 교체.
#
# 배경: literature_1500_intersection(leaky, ~60%/fold) vs pdac_consistency_1500(leak-free,
# 외부 5개 데이터셋 교차분석)을 5seed(84,126,168,210,252)로 비교한 결과 — internal은 유의하게
# literature_1500이 높았지만(p=0.017) external은 방향은 같아도 비유의(p=0.19). internal 우위가
# leakage 아티팩트일 가능성과 진짜 신호일 가능성이 둘 다 있어 판단이 갈렸다.
#
# 사용자 제안: pdac_consistency_1500(외부 검증, leakage 없음)과 TCGA-only Cox 회귀(nominal
# p<0.01, 719개, CPTAC 전혀 미참조 — literature_1500_intersection의 leakage 원인이던
# "CPTAC 라벨이 섞여 들어간다"는 성격 자체가 없음)를 합쳐서, 외부 검증 + 코호트 고유 신호를
# 동시에 반영하면서도 CPTAC 쪽 leakage는 원천 차단한 유전자셋을 새로 만들자는 것.
# 교집합(147개)은 기존 대비 너무 작아져서 합집합(2072개)으로 확정(data/dataset.py::
# pdac_consistency_cox_union_gene_ids).
#
# 처음부터 5seed로 돌린다 — 2seed로 먼저 보고 나중에 늘리는 과정 자체가 이번 pdac_consistency
# 비교에서 결론을 두 번 뒤집은 원인이었다(2seed일 땐 "차이 없다"였다가 5seed에서 유의한 internal
# 차이가 드러남) — 이번엔 처음부터 통계적으로 믿을만한 시드 수로 시작.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PORPOISE_uni2native_PDACCOXUNION_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,168,210,252 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PORPOISE_uni2native_PDACCOXUNION_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126,168,210,252 --n-folds 5 --bootstrap 2000
# literature_1500/pdac_consistency_1500 두 태그와 각각 paired bootstrap 비교(--model-a/-b만
# 바꿔서, scripts/paired_bootstrap_delta.py).
#
# 제출: sbatch sbatch/porpoise_final_pdaccoxunion_5seed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84 126 168 210 252)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2native_PDACCOXUNION_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1_kfold5_fold${FOLD}.log"

echo "=== PORPOISE final recipe(pdac_consistency_cox_union_tcga) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes pdac_consistency_cox_union_tcga --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906porpoise_final_pdaccoxunion_5seed_kfold5_array 2>&1 | tee "${log}"
echo "=== PORPOISE final recipe(pdac_consistency_cox_union_tcga) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
