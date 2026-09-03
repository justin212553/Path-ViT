#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M4-litcat8
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_litcat8_stg_kfold_array_%a.log

# 2026-09-03: BRCA M4 RNA 패널을 PAAD의 pathway8과 정확히 같은 원칙(문헌 큐레이션 유전자를
# 개별로 안 쓰고 8개 생물학적 카테고리 평균으로 압축)으로 맞춰서 재검증한다 — 사용자 지시:
# "Baseline인 PADC에서 뭉치기로 했으면 BRCA에서도 뭉치자. 이걸로 BRCA M4도 테스트하고. 만약
# 여기서 유의가 나온다, 그럼 PADC에서의 실험들은 '코호트가 작아서 신호가 안 난 거지 모델
# 자체는 건재하다'라고 퉁치고 넘어가자고."
#
# --gene-selection literature_categorized (scripts/brca_common.py::load_literature_categories/
# load_rna_matrix_categorized, PAM50+Oncotype DX+pan-cancer 6개 = 정확히 8개 카테고리, pathway8과
# 카테고리 개수 일치) + --clinical-staging. brca_m4_coxfdr_stg_kfold_array_hpc.sh와 동일하게
# institution(BH) external holdout은 안 쓴다(--external-tss none, 2026-08-31 결정).
#
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5. seed=84/126, fold=0..4.
#
# 완료 후(10개 fold 로그 확인, .logs/kfold_preds/에 CSV 10개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PMA_LITCAT8_STG_SS_AUX --seeds 84,126 --n-folds 5 --bootstrap 2000
#   M7과의 paired 비교(scripts/paired_bootstrap_delta.py --split internal)는 동일
#   --gene-selection literature_categorized --clinical-staging --external-tss none로 돌린 M7
#   k-fold CSV가 있어야 함 — scripts/run_brca_m7_litcat8_kfold_local.sh 참조.
#
# 제출: sbatch sbatch/brca_m4_litcat8_stg_kfold_array_hpc.sh

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

log=".logs/train_brca_m4_litcat8_stg_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M4 litcat8+stg seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection literature_categorized --clinical-staging \
    --external-tss none --group-ts 0903_brca_m4_litcat8_stg_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M4 litcat8+stg seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
