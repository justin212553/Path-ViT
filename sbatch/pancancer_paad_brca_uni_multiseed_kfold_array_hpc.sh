#!/bin/bash
#SBATCH --job-name=PVT-pancancer-paad-brca-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pancancer_paad_brca_multiseed_kfold_array_%a.log

# scripts/train_pancancer_paad_brca.py — PAAD(TCGA-PAAD)+BRCA(TCGA-BRCA) 공동학습(WSI trunk
# weight-tying, UNI v1 backbone). 로컬 seed=84 단일 파일럿(2026-08-11)에서 internal
# 0.6911(best-ckpt)/0.7602(final), external 0.6407/0.6513 — PAAD 단독 baseline(UNI2 기준
# internal 0.6359/external 0.6337, 3seed 앙상블) 대비 뚜렷한 개선. 이번 세션에서 나온
# 개입들이 거의 다 internal/external 트레이드오프였는데 이건 둘 다 올라서 재현 여부를
# 3seed(42/84/126)x5fold로 검증한다.
#
# [사전 준비 필수] BRCA 데이터(patches_tcga_brca/tiles/*/{coords.pt,features_uni.pt} ~47GB,
# brca_clinical.csv, brca_slide_manifest.csv, rna_brca.csv,
# brca_rna_gene_selection/selected_genes_top_1500.csv)가 HPC에 원래 전혀 없었다 — 로컬에서
# `python -m scripts.zip_brca_for_hpc`로 압축해 옮긴 뒤 `unzip -o brca_for_hpc.zip`로
# 프로젝트 루트에 풀어야 이 스크립트가 돌아간다. PAAD 쪽(features_uni.pt)은 이전 backbone
# 비교 작업 때 이미 HPC에 있을 가능성이 높지만, 없으면 --dataset tcga용 uni feature도
# 별도로 확인 필요.
#
# SLURM_ARRAY_TASK_ID(0~14) -> seed_idx = id/5, fold = id%5 (다른 multiseed array와 동일 관례).
# BRCA는 이 실험에서 held-out 평가 대상이 아니므로 fold 개념 없이 --seed 기준 train 부분만 쓴다.
#
# 완료 후(15개 fold 로그 확인, .logs/kfold_preds/·.logs/external_preds/에 CSV 15개씩 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PANCANCER_PAAD_BRCA_uni_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PANCANCER_PAAD_BRCA_uni_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/pancancer_paad_brca_uni_multiseed_kfold_array_hpc.sh

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

log=".logs/train_pancancer_paad_brca_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== pancancer PAAD+BRCA seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_pancancer_paad_brca --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --group-ts 0811pancancer_paad_brca_multiseed_hpc 2>&1 | tee "${log}"
echo "=== pancancer PAAD+BRCA seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
