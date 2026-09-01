#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-noaux-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_no_aux_multiseed_kfold_array_%a.log

# ViT_PORPOISE(models/vit_porpoise.py, --PORPOISE) — 지금까지 나온 모든 변형(plain gated-ABMIL/
# mean-pool/RNA co-attention, 나이스트롬 유무, dispersion/rna-aux 유무) 중 seed84/fold0
# internal C가 가장 높았던 "no_aux" 조합(dispersion 유지, rna-aux 제거, plain gated-ABMIL,
# BilinearFusion, 나이스트롬 유지 — sbatch/porpoise_uni2_stg_r_ablation_seed84_fold0_hpc.sh의
# array index 1, C=0.7119, HR=3.440, log-rank p=0.0154 유일하게 유의)를 3seed(42/84/126)×5fold
# 멀티시드 kfold로 검증한다. 단일 fold 결과는 이 프로젝트에서 반복적으로 재현 안 됐던 전례가
# 많아(findings_backlog.md), 이 architecture family가 진짜 M7(RNA+Clinical only, WSI 없음,
# 2seed×5fold pooled internal 0.6552/external 0.6221, paper/final_results_summary.md)를
# 통계적으로 유의미하게 이기는지 확정하는 게 이 job의 목적.
#
# M4A array(sbatch/m4a_uni2_coxadd_stg_r_multiseed_kfold_array_hpc.sh)와 동일 관례 —
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5.
#
# 완료 후(15개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 15개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PORPOISE_uni2_INT1500_SS_STG_R_DISP --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 그다음 external 평가는 sbatch/porpoise_no_aux_multiseed_external_eval_hpc.sh(재학습 없이
# eval-external-ckpt로 checkpoint 재사용, 이 15개 학습이 전부 끝난 뒤 제출).
#
# 제출: sbatch sbatch/porpoise_no_aux_multiseed_kfold_array_hpc.sh

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

log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2_INT1500_SS_STG_R_DISP_kfold5_fold${FOLD}.log"

echo "=== PORPOISE no_aux(uni2,STG+R,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes literature_1500_intersection --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 \
    --clinical-margin --clinical-staging \
    --patch-keep-frac 0.8 --attn-dispersion \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0831porpoise_no_aux_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== PORPOISE no_aux(uni2,STG+R,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
