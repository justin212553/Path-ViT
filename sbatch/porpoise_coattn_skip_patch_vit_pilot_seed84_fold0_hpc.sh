#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-coattn-novit
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_coattn_skip_patch_vit_pilot_seed84_fold0.log

# 독립 실험(PORPOISE 쪽 --porpoise-meanpool 검증과 별개, 병행 진행) — "나이스트롬이 patch 간
# 차이를 뭉개서 그 뒤의 RNA co-attention이 구별을 못 하는 것 아니냐"는 가설을, concat
# fusion(M4A 단독, sbatch/m4a_skip_patch_vit_pilot_seed84_fold0_hpc.sh — 이건 그대로 별도
# 진행)뿐 아니라 **BilinearFusion(Kronecker product)과 결합해서도** 검증한다.
#
# --PORPOISE --porpoise-coattn: attn_pool을 models/vit_m4a.py::CoAttentionPooling(M4A와
# 동일, RNA가 query인 cross-attention)으로 교체 — PORPOISE 원래의 RNA-무관 gated-ABMIL이
# 아니라 다시 RNA-guided로 되돌린 조합.
# --skip-patch-vit: 나이스트롬을 건너뛰고 CNN에서 나온 patch feature를 그대로 co-attention에
# 먹인다 — 지금까지 모든 co-attention 실험(M4A/MCAT/PMA)은 나이스트롬을 거친 토큰 위에서만
# 돌았다.
#
# 비교 기준: PORPOISE no_aux(dispersion 유지, aux 제거, plain gated-ABMIL, C=0.7119) 및
# M4A 기존 파일럿(co-attention, 나이스트롬 있음, C=0.61). rna-aux-weight는 no_aux 조합과
# 동일하게 뺐다(aux가 PORPOISE에서 소폭 해로운 것으로 나온 전례, 2026-08-31 ablation).
#
# 제출: sbatch sbatch/porpoise_coattn_skip_patch_vit_pilot_seed84_fold0_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

echo "=== PORPOISE coattn+skip-patch-vit(no_aux 레시피) seed=84 fold=0/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --porpoise-coattn --skip-patch-vit \
    --rna-genes literature_1500_intersection --dataset tcga --seed 84 \
    --backbone uni2 \
    --clinical-margin --clinical-staging \
    --patch-keep-frac 0.8 --attn-dispersion \
    --fold 0 --n-folds 5 --group-ts 0831porpoise_coattn_novit
echo "=== PORPOISE coattn+skip-patch-vit(no_aux 레시피) seed=84 fold=0/5 Complete: $(date) ==="
