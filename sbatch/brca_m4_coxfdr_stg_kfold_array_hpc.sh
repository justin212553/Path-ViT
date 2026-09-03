#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M4-coxfdr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_coxfdr_stg_kfold_array_%a.log

# 2026-09-03: brca_m4_multiseed_kfold_array_hpc.sh(고분산 RNA 패널, TOP1500)의 변형 —
# 로컬 5시드 M7 비교(scripts/run_brca_m7_gene_selection_compare_5seed_local.sh)에서 생존
# 라벨 기반 Cox 유전자 선정(특히 BH-FDR q<0.1로 잡음 필터링한 121개 패널)이 기존 고분산
# 패널보다 internal c-index가 뚜렷이 높고(0.748 vs 0.657, 5시드 평균) external도 밀리지
# 않는다는 게 확인됨(사용자 승인 후 M4로 확대) — 여기에 T/N/M staging도 함께 켠다
# (--clinical-staging, 2026-09-02 GDC 추출 버그 수정 후 새로 추가된 기능).
#
# brca_m4_multiseed_kfold_array_hpc.sh와 동일하게 institution(BH) external holdout은 안 쓴다
# (--external-tss none, 2026-08-31 결정 — BH 인구가 event rate 등에서 나머지와 너무 달라
# 신뢰하기 어렵다고 판단) — 1058명 전체를 internal k-fold 풀로 쓴다.
#
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5. seed=84/126, fold=0..4.
#
# 완료 후(10개 fold 로그 확인, .logs/kfold_preds/에 CSV 10개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PMA_FDR0.1_COXGENE_STG_SS_AUX --seeds 84,126 --n-folds 5 --bootstrap 2000
#   M7과의 paired 비교(scripts/paired_bootstrap_delta.py)는 동일 --gene-selection cox
#   --fdr-threshold 0.1 --clinical-staging --external-tss none로 돌린 M7 k-fold CSV가 있어야
#   함 — scripts/run_brca_m7_coxfdr_stg_kfold_local.sh 참조.
#
# 제출: sbatch sbatch/brca_m4_coxfdr_stg_kfold_array_hpc.sh

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

log=".logs/train_brca_m4_coxfdr_stg_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M4 coxfdr+stg seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection cox --fdr-threshold 0.1 --clinical-staging \
    --external-tss none --group-ts 0903_brca_m4_coxfdr_stg_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M4 coxfdr+stg seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
