#!/bin/bash
#SBATCH --job-name=PVT-M4-relbias-pdaccons
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m4_relbias_pdaccons1500_cnv_mut_clr100_kfold_array_%a.log

# 2026-09-04: "나이스트롬 대신 진짜 dense full self-attention을 써봤어야 하지 않았나"(사용자
# 질문)에 대한 답 — models/vit_encoder.py::RelativeBiasFullAttention(--rel-bias-attention,
# Swin류 학습되는 상대좌표 attention bias를 곁들인 O(N^2) full self-attention, 2026-07-23에
# 이미 구현은 돼 있었음)을 PAAD(패치 수<=544, dense attention 부담 없다고 이미 확인됨)에서
# 오늘 채택한 M4 baseline(pdac_consistency_1500+CNV+mutation+STG+margin+CLR100+uni2native)
# 위에 얹어 처음으로 끝까지 돌린다. --rel-bias-attention을 켜면 config.py 관례상 use_nystrom/
# use_spatial_embed가 자동으로 False로 강제된다(ViT_M1.__init__) — 별도 플래그 불필요.
#
# 참고(2026-07-22/23, 같은 계열의 이전 실험, findings_backlog.md/config.py 주석):
#   - KNNBiasAttention(--knn-bias-attention, 국소 attention만) 단독은 PAAD에서 이미 더 나빴음
#     (internal 0.6309->0.6094, external 0.6289->0.5880).
#   - RelativeBiasFullAttention 자체는 BRCA에서 돌리다 패치 수(중앙값 10,309)가 너무 많아
#     즉시 CUDA OOM났던 기록만 로컬에 남아있고, PAAD에서 끝까지 돌려 나온 수치는 없었음 —
#     이번이 그 공백을 처음 메우는 실행.
#
# 오늘 채택한 baseline(비교 기준, 2seed x 5fold pooled, RELBIAS 없음):
#   internal c-index = 0.600 (p=0.163), external c-index = 0.639 (p=0.0008)
#
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5. seed=84/126, fold=0..4.
# --requeue: free-gpu partition preemption 대비.
#
# 완료 후(10개 fold 로그 확인, .logs/kfold_preds/에 CSV 10개 있는지 확인 — 정확한 --model 태그는
# 로그 상단의 "Model:" 출력 또는 .logs/kfold_preds/ 파일명에서 직접 확인할 것, 대략
# M4_uni2native_PDACCONS1500_CNV_STG_R_MUT_RELBIAS_COX_ADD_CLR100_LRMW10 형태로 예상):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model <위에서 확인한 태그> --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model <위에서 확인한 태그> --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m4_relbias_pdaccons1500_cnv_mut_clr100_kfold_array_hpc.sh

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

log=".logs/train_tcga_m4_relbias_pdaccons1500_cnv_mut_clr100_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M4(RELBIAS, pdac_consistency_1500+CNV+mutation+STG+R+CLR100, uni2native) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native --rna-genes pdac_consistency_1500 --use-cnv --clinical-mutation \
    --clinical-staging --clinical-margin --combine-mode cox_add \
    --rel-bias-attention \
    --clinical-lr-mult 100 --lr-mult-warmup-epochs 10 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --group-ts 0905_m4_relbias_pdaccons1500_cnv_mut_clr100_kfold_array 2>&1 | tee "${log}"
echo "=== M4(RELBIAS, pdac_consistency_1500+CNV+mutation+STG+R+CLR100, uni2native) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
