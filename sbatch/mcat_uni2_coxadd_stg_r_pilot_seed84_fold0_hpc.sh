#!/bin/bash
#SBATCH --job-name=PVT-MCAT-uni2-coxadd-stg-pilot
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/mcat_uni2_coxadd_stg_r_pilot_seed84_fold0.log

# ViT_MCAT(models/vit_mcat.py, --MCAT) 첫 HPC 파일럿 — 2026-08-31 "다른 SOTA 모델들 한번 찾아보자"
# 조사에서 나온 3단계 계획의 Phase 1: RNA를 PDAC 기능별 8개 유전자 카테고리(GeneGroupEncoder,
# 학습되는 선형결합, pathway8의 unsigned-mean 실패를 구조적으로 피함)로 나눠 만든 8개 pathway
# 토큰이 patch 전체에 동시에 co-attention한다(진짜 MCAT/SurvPath 스타일, M4A의 단일-query
# 버전을 확장). 목표: attention entropy 붕괴(query 1개 vs key 4개의 저용량 co-attention,
# findings_backlog.md 최상위 발견) → query 개수를 늘려 정면 해소가 되는지 1-fold pilot으로
# 먼저 방향성만 본다(성공하면 Self-Enhancement Learning 보조 loss, 그래도 안 되면 PORPOISE로).
#
# M4/M4A/PMA와 동일한 최종 레시피(uni2, cox_add, staging, margin, attn-dispersion,
# patch-keep-frac 0.8, rna-aux-weight 1.0)로 seed=84 fold=0 딱 1개만 돌려, 기존에 이미 갖고
# 있는 M1/M2/M4/M4A/PMA seed84 fold0 파일럿 숫자(internal C: 0.68/0.61/0.66 등)와 바로
# 비교 가능하게 한다. 여기서 뚜렷한 개선이 없으면 멀티시드 kfold array로 확장할 필요도 없다.
#
# 로컬에서 --epochs 1 smoke test는 이미 통과(구조/gradient 흐름/체크포인트 태깅 확인 완료) —
# 이 job이 실제 학습 신호가 나오는 첫 실행.
#
# 제출: sbatch sbatch/mcat_uni2_coxadd_stg_r_pilot_seed84_fold0_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

echo "=== MCAT(uni2,cox_add,STG+R) seed=84 fold=0/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --MCAT --rna-genes literature_1500_intersection --dataset tcga --seed 84 \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --fold 0 --n-folds 5 --group-ts 0831mcat_uni2_coxadd_stg_r_pilot
echo "=== MCAT(uni2,cox_add,STG+R) seed=84 fold=0/5 Complete: $(date) ==="
