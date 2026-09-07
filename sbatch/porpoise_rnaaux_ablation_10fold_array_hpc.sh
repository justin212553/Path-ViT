#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-rnaaux-10fold
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_rnaaux_ablation_10fold_array_%a.log

# 2026-09-06: sbatch/porpoise_both_loss_10fold_array_hpc.sh(현재 확정된 최종 레시피 —
# literature_1500_intersection, uni2native, CLR100+CNV+mutation, attn-dispersion, surv-loss
# both weight=1.0)에 --rna-aux-weight 1.0(RNAPredictionHead 보조과제 — WSI 임베딩이 RNA
# 발현값도 같이 예측하도록 시킴)만 추가한 ablation. PORPOISE 공식 코드엔 없는 개념(PMA
# 레시피에서만 써오던 것)이고, PAAD의 예전 "no_aux" 레시피 선정 자체가 "이걸 넣으면 오히려
# 소폭 해로웠다"는 결론에서 나온 거라 효과가 없을 걸로 예상되지만 — literature_1500 사양으로
# 직접 켜/끄기 비교 데이터를 남겨두기 위해 확인 차 돌린다(사용자 지시).
#
# 나머지 레시피는 porpoise_both_loss_10fold_array_hpc.sh와 완전히 동일 — --rna-aux-weight
# 유무 하나만 다르다.
#
# 완료 후, --rna-aux-weight 없는 버전과 paired bootstrap 직접 비교:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PORPOISE_uni2native_INT1500_CNV_SS_AUX_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PORPOISE_uni2native_INT1500_CNV_SS_AUX_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/paired_bootstrap_delta.py --split internal --dataset tcga \
#       --model-a PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --model-b PORPOISE_uni2native_INT1500_CNV_SS_AUX_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/paired_bootstrap_delta.py --split external --dataset cptac \
#       --model-a PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --model-b PORPOISE_uni2native_INT1500_CNV_SS_AUX_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/porpoise_rnaaux_ablation_10fold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2native_INT1500_CNV_SS_AUX_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1_kfold5_fold${FOLD}.log"

echo "=== PORPOISE both loss + RNA AUX(literature_1500) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906porpoise_rnaaux_2seed_kfold5_array 2>&1 | tee "${log}"
echo "=== PORPOISE both loss + RNA AUX(literature_1500) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
