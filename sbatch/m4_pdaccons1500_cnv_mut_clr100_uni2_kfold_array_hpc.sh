#!/bin/bash
#SBATCH --job-name=PVT-M4-pdaccons-uni2
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m4_pdaccons1500_cnv_mut_clr100_uni2_kfold_array_%a.log

# 2026-09-05: m4_pdaccons1500_cnv_mut_clr100_uni2native_kfold_array_hpc.sh(정식 채택 backbone,
# 256px@0.5MPP, UNI2-h 공식 학습 스펙 그대로)의 external이 M7보다 오히려 낮게 나온 것(delta
# -0.0544, p=0.076)을 보고 사용자가 던진 질문 — "이게 backbone 정체성 문제가 아니라 단순
# 해상도(패치 물리적 크기) 문제 아닐까?" — uni2native는 128um x 128um로 병리의 저배율 시야보다
# 훨씬 좁다(2026-09-04 논의). 이 스크립트는 그 대조군이다:
#
#   uni2(이 스크립트)  : 1024px 타일(우리 resnet50과 동일 물리 크기 1024um x 1024um, data/
#                        extract_features.py)을 512px로 리사이즈해 UNI2-h에 입력 — UNI2-h
#                        자체의 공식 학습 스펙(256px@20x/0.5um)과는 안 맞는 해상도 불일치가
#                        있지만, 물리적으로 uni2native보다 8배 넓은 시야를 담는다.
#   uni2native(비교 대상): 256px@0.5MPP, UNI2-h 공식 스펙과 정확히 일치 — 해상도 불일치는
#                        없지만 물리적 시야가 uni2/resnet50의 1/64(면적 기준).
#
# 만약 uni2(해상도 불일치가 있는데도 시야가 넓은 쪽)가 uni2native보다 M4-M7 delta가 덜
# 나쁘게 나온다면, "backbone을 뭘 쓰느냐"가 아니라 "패치 물리적 크기가 문제"라는 가설에
# 힘이 실린다 — 사용자 표현대로 "경종을 울릴" 결과. 레시피는 uni2native 스크립트와 완전히
# 동일, --backbone만 다르다.
#
# 비교 기준(2026-09-05 확인, 같은 레시피의 uni2native/resnet50 결과):
#   M7 baseline:            internal C=0.5988, external C=0.6291
#   M4(uni2native, 정식):    internal C=0.5985 (delta -0.0003, p=0.989), external C=0.5747 (delta -0.0544, p=0.076)
#   M4(resnet50, 사고):      internal C=0.5997 (delta +0.0009, p=1.000), external C=0.6385 (delta +0.0094, p=0.676)
#
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5. seed=84/126, fold=0..4.
# --requeue: free-gpu partition preemption 대비.
#
# 완료 후(10개 fold 로그 확인, .logs/kfold_preds/·.logs/external_preds/에 CSV 10개씩 있는지 확인
# — 2026-09-05 train.py 수정으로 k-fold 모드에서도 external CSV가 이제 자동 저장됨):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model M4_uni2_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_CLR100_LRMW10 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model M4_uni2_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_CLR100_LRMW10 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/paired_bootstrap_delta.py --split internal --dataset tcga \
#       --model-a M7_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_CLR100 \
#       --model-b M4_uni2_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_CLR100_LRMW10 --seeds 84,126 --n-folds 5 --bootstrap 2000
#   python scripts/paired_bootstrap_delta.py --split external --dataset cptac \
#       --model-a M7_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_CLR100 \
#       --model-b M4_uni2_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_CLR100_LRMW10 --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/m4_pdaccons1500_cnv_mut_clr100_uni2_kfold_array_hpc.sh

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

log=".logs/train_tcga_m4_pdaccons1500_cnv_mut_clr100_uni2_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== M4(pdac_consistency_1500+CNV+mutation+STG+R+CLR100, uni2/resize-mismatch) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4 --dataset tcga --external --seed "${SEED}" \
    --backbone uni2 --rna-genes pdac_consistency_1500 --use-cnv --clinical-mutation \
    --clinical-staging --clinical-margin --combine-mode cox_add \
    --clinical-lr-mult 100 --lr-mult-warmup-epochs 10 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --group-ts 0905_m4_pdaccons1500_cnv_mut_clr100_uni2_kfold_array 2>&1 | tee "${log}"
echo "=== M4(pdac_consistency_1500+CNV+mutation+STG+R+CLR100, uni2/resize-mismatch) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
