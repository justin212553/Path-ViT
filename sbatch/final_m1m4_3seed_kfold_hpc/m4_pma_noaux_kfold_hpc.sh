#!/bin/bash
#SBATCH --job-name=PVT-M4PMA-NOAUX-2seed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/m4_pma_noaux_kfold_array_%a.log

# M4/PMA ablation — RNA 예측 보조과제(--rna-aux-weight, models/rna_predictor.py::
# RNAPredictionHead)를 끈 효과만 분리해서 본다. m4_pma_3seed_kfold_array_hpc.sh(최종 채택
# 레시피)에서 --rna-aux-weight 1.0 한 줄만 빼고 나머지(backbone/clinical/combine-mode/
# attn-dispersion)는 전부 동일 — attn-dispersion은 이 ablation에서는 그대로 켜둔다(RNA AUX와
# attn-dispersion을 동시에 바꾸면 어느 쪽 효과인지 못 가르니 한 번에 하나씩).
#
# 최종표(paper/results_table_pma_family_3seed_kfold_ci.md)와 동일하게 seed 84/126만 사용
# (seed42는 WSI 모델에 불리한 이상치로 최종 채택에서 제외된 시드 — 이 ablation도 같은 기준으로
# 비교해야 공정).
#
# 태그: PMA_uni2native_INT1500_SS_STG_R_DISP_COX_ADD (기존 _AUX만 빠짐 — train.py model_prefix
# 부착 순서상 rna_aux_weight>0일 때만 _AUX가 붙으므로 자동으로 빠진다).
#
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5.
#
# 완료 후:
#   1. 일반 --fold 경로는 external CSV를 안 남기므로 eval-external-ckpt 스윕 먼저(HPC에서):
#        sbatch sbatch/final_m1m4_3seed_kfold_hpc/eval_external_ckpt_sweep_hpc.sh
#      (scripts/final_eval_external_ckpt_sweep.py에 M4_noaux 항목 등록해둠 — 다른 모델도 같이
#      재실행되지만 결과는 그대로라 무해)
#   2. 풀링:
#        python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PMA_uni2native_INT1500_SS_STG_R_DISP_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000
#        python scripts/pool_multiseed_external_preds.py --dataset cptac --model PMA_uni2native_INT1500_SS_STG_R_DISP_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000
#   3. 원래 M4(AUX 있음, internal=0.6488/external=0.6370, paper/results_table_...md 참조)와
#      비교 — paired bootstrap이 필요하면:
#        python scripts/paired_bootstrap_delta.py --split internal --dataset tcga --model-a PMA_uni2native_INT1500_SS_STG_R_DISP_COX_ADD --model-b PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 84,126
#        python scripts/paired_bootstrap_delta.py --split external --dataset cptac --model-a PMA_uni2native_INT1500_SS_STG_R_DISP_COX_ADD --model-b PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 84,126
#      (--pred-root 없이 .logs를 그대로 쓴다 — paper/final_preds_snapshot에는 noaux 결과가 없음)
#
# 제출: sbatch sbatch/final_m1m4_3seed_kfold_hpc/m4_pma_noaux_kfold_hpc.sh

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

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

log="paper/.hpc/train_tcga_seed${SEED}_PMA_uni2native_INT1500_SS_STG_R_DISP_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== M4/PMA-NOAUX(WSI+Clinic+RNA,cox_add,RNA-aux 없음) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --dataset tcga --external --seed "${SEED}" \
    --rna-genes literature_1500_intersection \
    --backbone uni2native \
    --clinical-staging --clinical-margin \
    --patch-keep-frac 0.8 --attn-dispersion \
    --combine-mode cox_add \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0830_m4pma_noaux_2seed_kfold_hpc 2>&1 | tee "${log}"
echo "=== M4/PMA-NOAUX(WSI+Clinic+RNA,cox_add,RNA-aux 없음) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
