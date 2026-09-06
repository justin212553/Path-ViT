#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-bothloss-10fold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_both_loss_10fold_array_%a.log

# 2026-09-06: cox vs nll_surv 매칭 비교(sbatch/porpoise_cox_loss_matched_10fold_array_hpc.sh vs
# porpoise_nll_surv_loss_10fold_array_hpc.sh) 결과 — 둘이 서로 다른 걸 최적화하는 것으로
# 보였다: nll_surv는 raw C-index가 근소 우세(internal 0.6515 vs 0.6382)인데, cox는 HR/log-rank
# 유의성(p=0.0021 vs 0.089, cox만 유의)과 seed 간 안정성(std 0.0104 vs 0.0232)이 확실히
# 우세했다. Cox score test가 log-rank와 점근적으로 동일하다는 사실과 정합적 — nll_surv는
# 전체 우도(hazard curve 형태)를 맞추는 데 최적화되고, 이분화 분리력을 직접 밀지는 않는다.
#
# --surv-loss both(train.py 2026-09-06 신규): 같은 risk_head hazard logit에서
# nll_surv_loss와 (utils/losses.py::hazard_to_risk로 유도한 스칼라에) cox_ph_loss를 같이
# 계산해서 더한다(가중치 --nll-cox-weight, 기본 1.0 동등가중) — 두 loss가 서로 다른 강점을
# 보완할 수 있는지 보는 ablation. 나머지 레시피는 cox/nll_surv 매칭 실험과 완전히 동일
# (uni2native, CLR100, CNV, mutation, literature_1500_intersection).
#
# 완료 후, cox/nll_surv 두 태그와 함께 3-way paired bootstrap 비교:
#   python scripts/paired_bootstrap_delta.py --split internal --dataset tcga \
#       --model-a PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100 \
#       --model-b PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
# (외부/nll_surv 단독과의 비교도 --model-a/--model-b만 바꿔 동일하게 실행)
#
# 제출: sbatch sbatch/porpoise_both_loss_10fold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1_kfold5_fold${FOLD}.log"

echo "=== PORPOISE both loss(nll_surv+cox, uni2native,CLR100,CNV,MUT) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906porpoise_bothloss_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== PORPOISE both loss(nll_surv+cox, uni2native,CLR100,CNV,MUT) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
