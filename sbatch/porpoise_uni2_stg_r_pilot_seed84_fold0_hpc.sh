#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-uni2-pilot
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_uni2_stg_r_pilot_seed84_fold0.log

# ViT_PORPOISE(models/vit_porpoise.py, --PORPOISE) 첫 HPC 파일럿 — 3단계 계획의 Phase 3.
# Phase 1(MCAT 스타일 multi-pathway co-attention)까지 포함해 M4/M4A/PMA/MCAT 전부 "attention이
# 중요 patch를 찾아내야 하는" 계열이었고, query를 1개→8개로 늘려도 co-attention entropy가
# 0.9998(uniform)로 붕괴하는 게 gradient 부족과도 무관하다는 게 확인됐다(findings_backlog.md
# 2026-08-31 최상위 발견, scripts/diagnose_mcat_gradients.py). PORPOISE는 WSI를 RNA와 무관한
# 평범한 gated-ABMIL로 풀링하고, WSI-RNA 상호작용은 풀링 이후 Kronecker/bilinear product로
# 명시적으로 포착한다 — attention이 patch를 구별할 필요가 아예 없는 구조.
#
# --combine-mode는 안 줌(ViT_PORPOISE가 내부적으로 항상 cox_add로 고정, models/vit_porpoise.py
# 참조 — clinical은 항상 별도 Cox 가산항). M4A/MCAT과 같은 최종 레시피 나머지는 그대로 맞춰
# seed84 fold0 internal C를 바로 비교 가능하게 한다(PMA 0.68 / PM4 0.66 / M4A 0.61 / MCAT 0.46).
#
# 로컬에서 --epochs 1 smoke test는 이미 통과(구조/gradient 흐름/체크포인트 태깅 확인 완료) —
# 이 job이 실제 학습 신호가 나오는 첫 실행.
#
# 제출: sbatch sbatch/porpoise_uni2_stg_r_pilot_seed84_fold0_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

echo "=== PORPOISE(uni2,STG+R) seed=84 fold=0/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes literature_1500_intersection --dataset tcga --seed 84 \
    --backbone uni2 \
    --clinical-margin --clinical-staging \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --fold 0 --n-folds 5 --group-ts 0831porpoise_uni2_stg_r_pilot
echo "=== PORPOISE(uni2,STG+R) seed=84 fold=0/5 Complete: $(date) ==="
