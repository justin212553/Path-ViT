#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M4-cons
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_consistency_stg_kfold_array_%a.log

# 2026-09-04: brca_m4_coxfdr_stg_kfold_array_hpc.sh의 변형 — RNA 패널을 생존 라벨 기반(Cox+FDR)
# 대신 "완전히 객관적인" 새 패널(scripts/select_brca_rna_genes_consistency.py, 882유전자)로
# 교체한다. PDAC의 pdac_consistency_1500과 정확히 같은 설계 철학:
#   - Győrffy(2021, CSBJ) 55개 독립 GEO 유방암 데이터셋(7830명) 통합 생존분석
#     (relapse-free survival 기준 Cox+FDR, 우리 TCGA-BRCA 코호트는 전혀 미참조)
#   - + 기존 BRCA 도메인 문헌(PAM50+Oncotype DX+pan-cancer 6카테고리, 165유전자)
#   - + PDAC_LITERATURE_GENE_SETS의 "core_driver_tumor_suppressor"(TP53/PIK3CA/PTEN/MYC/BRAF
#     등 범암종 드라이버, 이전 BRCA 문헌 패널에서 빠져 있었음 — 사용자 지적으로 추가)
# variance_1500(고분산, 우리 코호트 통계 기반)과 달리 어떤 코호트/라벨도 안 봐서 완전히
# leak-free — "다른 데이터셋에서 추출한 리스트로 비교해야 공정하다"(사용자 지시)는 요구를 충족.
#
# --external-tss none(2026-08-31 결정과 동일) — 1058명 전체를 internal k-fold 풀로 씀.
#
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5. seed=84/126, fold=0..4.
# --requeue: free-gpu partition preemption 대비(2026-09-03 LITCAT8 seed84가 이걸로 두 번 끊긴 전례).
#
# 완료 후(10개 fold 로그 확인, .logs/kfold_preds/에 CSV 10개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PMA_CONS882_STG_SS_AUX --seeds 84,126 --n-folds 5 --bootstrap 2000
#   M7과의 paired 비교(scripts/paired_bootstrap_delta.py)는 동일 --gene-selection consistency
#   --clinical-staging --external-tss none로 돌린 M7 k-fold CSV가 필요
#   (scripts/run_brca_m7_consistency_stg_kfold_local.sh 참조, M7은 로컬에서 돌림).
#
# 제출: sbatch sbatch/brca_m4_consistency_stg_kfold_array_hpc.sh

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

log=".logs/train_brca_m4_consistency_stg_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M4 consistency+stg seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --gene-selection consistency --clinical-staging \
    --external-tss none --group-ts 0904_brca_m4_consistency_stg_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M4 consistency+stg seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
