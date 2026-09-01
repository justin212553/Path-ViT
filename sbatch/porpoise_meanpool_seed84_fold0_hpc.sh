#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-meanpool
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_meanpool_seed84_fold0.log

# --porpoise-meanpool ablation — attn_pool(gated-ABMIL)을 무파라미터 MeanPooling으로 교체
# (models/vit_porpoise.py::MeanPooling). scripts/diagnose_porpoise_reliance.py로 확인한
# plain gated-ABMIL의 patch attention entropy가 0.999(거의 완전 uniform)였고, 이 붕괴가
# BRCA(N≈1058, findings_backlog.md 2026-07-22 heatmap 확인)에서도 재현됐다 — attention이
# "선택"하는 역할을 한 번도 못 해봤다는 뜻이라, 진짜 mean-pool로 바꿔도 성능이 같은지
# (=attention 모듈 자체가 불필요한지) 직접 검증한다.
#
# 레시피는 지금까지 가장 좋았던 no_aux 조합(dispersion 유지, aux 제거, seed84/fold0
# internal C=0.7119, 유일하게 log-rank 유의)에 --porpoise-meanpool만 추가 — 그 숫자와
# 바로 비교한다. 비슷하게 나오면 attention이 불필요하다는 게 실증되고, 그러면 다음
# 단계로 나이스트롬(--skip-patch-vit)도 제거해본다(사용자 확정).
#
# 제출: sbatch sbatch/porpoise_meanpool_seed84_fold0_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

echo "=== PORPOISE meanpool(no_aux 레시피) seed=84 fold=0/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --porpoise-meanpool --rna-genes literature_1500_intersection --dataset tcga --seed 84 \
    --backbone uni2 \
    --clinical-margin --clinical-staging \
    --patch-keep-frac 0.8 --attn-dispersion \
    --fold 0 --n-folds 5 --group-ts 0831porpoise_meanpool
echo "=== PORPOISE meanpool(no_aux 레시피) seed=84 fold=0/5 Complete: $(date) ==="
