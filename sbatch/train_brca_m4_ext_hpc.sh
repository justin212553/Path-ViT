#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M4-ext
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/train_brca_m4_ext_%j.log

# TCGA-BRCA로 M4(ViT_PMA, 최종 PAAD 레시피와 동일한 컴포넌트: patch-keep-frac 0.8, rna-aux-weight
# 1.0, backbone=uni — HF 사전추출 UNI v1 feature 그대로 씀, uni2/uni2native 아님, 2026-08-30
# 사용자와 확인함: BRCA는 UNI2로 사전추출된 적이 없고 raw WSI도 없어 이번 실험은 uni v1으로
# 진행)로 단발성(single seed, single run) 검증한다 — "BRCA는 데이터셋이 크니 여러 번 돌리지
# 말자"(사용자 지시)는 방침에 따라 3seed x 5fold 같은 반복 없이 딱 한 번만 돈다.
#
# institution-level external split: 가장 큰 단일 기관(BH, 공통 case 1058명 중 142명, 13.4%)을
# 통째로 external holdout(scripts/brca_common.py::EXTERNAL_TSS, 2026-08-30 추가)으로 뺀다 —
# TCGA-PAAD -> CPTAC-PDAC 구도를 단일 코호트인 BRCA에서 재현하기 위함. 나머지 916명을
# 6:2:2(train/val/test)로 internal split.
#
# RNA 유전자 리스트는 재선택 안 함(사용자 지시) — data/brca_rna_gene_selection/
# selected_genes_top_1500.csv(institution split 이전 기준으로 이미 고정된 파일)를 그대로 재사용.
#
# 목적: M4(WSI+Clinic+RNA)와 M7(scripts/train_brca_m7.py, Clinic+RNA)의 internal/external
# delta가 PAAD(paper/results_table_pma_family_3seed_kfold_ci.md 최종표: internal M4-M7=-0.0064,
# external M4-M7=+0.0149)와 유의미하게 다른지 1차 확인. 로컬 1epoch 스모크테스트로 코드 경로
# 정상 동작 확인 완료.
#
# 완료 후 CSV: .logs/kfold_preds/brca_BRCA_PMA_TOP1500_SS_AUX_EXTTSSBH_seed{SEED}.csv (internal),
#              .logs/external_preds/brca_BRCA_PMA_TOP1500_SS_AUX_EXTTSSBH_seed{SEED}.csv (external)
#
# 제출: sbatch sbatch/train_brca_m4_ext_hpc.sh
# (seed를 바꾸고 싶으면 아래 SEED= 한 줄만 수정)

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=84

echo "=== BRCA M4(ext=BH) seed=${SEED} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --group-ts 0830_brca_m4_ext_single
echo "=== BRCA M4(ext=BH) seed=${SEED} Complete: $(date) ==="
