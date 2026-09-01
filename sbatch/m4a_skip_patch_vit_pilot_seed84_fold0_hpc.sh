#!/bin/bash
#SBATCH --job-name=PVT-M4A-skippatchvit-pilot
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m4a_skip_patch_vit_pilot_seed84_fold0.log

# 독립 실험(PORPOISE 쪽 --skip-patch-vit 검증과 별개, 병행 진행) — "나이스트롬이 patch 간
# 차이를 뭉개서 그 뒤의 RNA co-attention이 구별을 못 하는 것 아니냐"는 가설을 M4A에서
# 직접 검증한다. 기존 M4A 파일럿(sbatch/m4a_uni2_coxadd_stg_r_multiseed_kfold_array_hpc.sh와
# 동일 레시피, seed84/fold0 internal C=0.61, --eval-internal-ckpt 진단에서 co-attention
# entropy 0.999+로 붕괴 확인됨)은 전부 --skip-patch-vit 없이(즉 나이스트롬을 거친 토큰 위에서)
# co-attention을 돌렸다 — RNA가 "원본 백본 feature"를 직접 본 적은 한 번도 없다.
#
# --skip-patch-vit만 추가해 나이스트롬을 완전히 건너뛰고, CNN에서 나온 patch feature를 그대로
# CoAttentionPooling(RNA query)에 먹인다. 나머지는 기존 M4A 최종 레시피와 동일 — internal
# C-index를 기존 M4A 0.61과 바로 비교 가능. entropy가 여전히 붕괴하면(가능성 높음, PORPOISE
# 쪽 plain-ABMIL에서도 나이스트롬 유무와 무관하게 0.999 나온 전례 참조) 나이스트롬은 원인이
# 아니었다는 뜻이고, 붕괴가 풀리면 나이스트롬이 실제로 patch 판별력을 죽이고 있었다는 뜻이 된다.
#
# 제출: sbatch sbatch/m4a_skip_patch_vit_pilot_seed84_fold0_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

echo "=== M4A skip-patch-vit(uni2,cox_add,STG+R) seed=84 fold=0/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --M4A --skip-patch-vit --rna-genes literature_1500_intersection --dataset tcga --seed 84 \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --fold 0 --n-folds 5 --group-ts 0831m4a_skip_patch_vit_pilot
echo "=== M4A skip-patch-vit(uni2,cox_add,STG+R) seed=84 fold=0/5 Complete: $(date) ==="
