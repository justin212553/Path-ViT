#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-genegroup-pilot
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_genegroup_pilot_seed84_kfold_array_%a.log

# 2026-09-06: 확정 최종 레시피(porpoise_final_pdaccons2000_2seed... 등과 동일 베이스)에서
# --rna-gene-groups 딱 하나만 추가한 파일럿(1시드, seed=84, 5fold). 유전자 인코딩 방식을
# flat RNAEncoder(모든 유전자를 하나의 MLP에 concat)에서 models/gene_group_rna_encoder.py::
# GeneGroupRNAEncoder(8개 PDAC 카테고리 + CNV를 9번째 카테고리로, 카테고리마다 독립 학습
# Linear로 토큰 생성 후 concat -> BilinearFusion)로 교체. 나머지(백본/RNA 유전자셋/CLR100/
# CNV/mutation/surv-loss/attn-dispersion)는 전부 동일.
#
# [주의 — 알려진 한계, 사용자 확인 후 진행] GeneGroupEncoder가 나누는 8개 카테고리 기준
# (PDAC_LITERATURE_GENE_SETS, 163개 고정 문헌 유전자)과 pdac_consistency_1500(JCI Insight
# 교차분석) 사이 겹치는 유전자가 66개뿐이다 — 카테고리에 없는 나머지 1434개 유전자는 이
# 인코더에서 조용히 안 쓰인다. 즉 이 파일럿은 "1500개를 카테고리로 재편성"이 아니라 "그중
# 66개짜리 부분집합을 카테고리별 학습 Linear로 인코딩"하는 실험이다(사용자 결정, 2026-09-06:
# "일단 66개로 파일럿해본다" — 카테고리를 pdac_consistency 전체를 덮게 새로 만드는 건 다음
# 단계로 미룸).
#
# 처음 돌리는 새 아키텍처 코드라 1시드(84)만, array도 fold 5개로 최소화 — 크래시 여부부터
# 확인 후 방향성 있으면 시드 확장.
#
# 완료 후:
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model PORPOISE_uni2native_PDACCONS1500_CNV_GENEGROUP_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84 --n-folds 5 --bootstrap 2000
#   python scripts/pool_multiseed_external_preds.py --dataset cptac \
#       --model PORPOISE_uni2native_PDACCONS1500_CNV_GENEGROUP_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1 \
#       --seeds 84 --n-folds 5 --bootstrap 2000
#
# 제출: sbatch sbatch/porpoise_genegroup_pilot_seed84_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=84
N_FOLDS=5
FOLD=$SLURM_ARRAY_TASK_ID

log=".logs/train_tcga_seed${SEED}_PORPOISE_uni2native_PDACCONS1500_CNV_GENEGROUP_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1_kfold5_fold${FOLD}.log"

echo "=== PORPOISE gene-group RNA encoder pilot seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes pdac_consistency_1500 --rna-gene-groups --dataset tcga --external --seed "${SEED}" \
    --backbone uni2native \
    --clinical-margin --clinical-staging \
    --clinical-lr-mult 100 --use-cnv --clinical-mutation \
    --patch-keep-frac 0.8 --attn-dispersion \
    --surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0906porpoise_genegroup_pilot_seed84_kfold5_array 2>&1 | tee "${log}"
echo "=== PORPOISE gene-group RNA encoder pilot seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
