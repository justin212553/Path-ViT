#!/bin/bash
#SBATCH --job-name=PVT-PMA-tileaug-disp
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_ex_ss_aux_tileaugment_dispersion_ext.log

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# scripts/_pma_ex_ss_aux_tileaugment_ext.sh(2026-07-21, dispersion 없음) + --attn-dispersion 추가.
# 그 스크립트는 이미 결과가 나온 실험 기록이라 그대로 두고, DISP를 더한 버전을 새로 만든다 —
# train_m1/m2/m3_hpc.sh가 이미 쓰고 있는 "SS+AUG+DISP(+AUX)" 레시피를 PMA(clinical 포함, M4
# 슬롯)에도 맞춘 것.
#
# --tile-decode-workers 8: models/vit_m1.py::_patch_tokens의 타일 디코딩+증강 스레드풀 크기.
# 이 작업은 forward() 안에서 도는 별도 스레드풀이라 DataLoader num_workers와는 무관하다 — 위
# --cpus-per-task=8을 실제로 다 쓰려면 이 값도 맞춰야 한다(2026-08-03, 그 전까지 4로 하드코딩).
#
# 제출: sbatch scripts/_pma_ex_ss_aux_tileaugment_dispersion_ext.sh

LogDir=".logs"
Seeds=(42 84 126)
GroupTs="0803pma_tileaugment_disp_ext"

for seed in "${Seeds[@]}"; do
    echo "=== PMA_EX_SS_AUX_AUG_DISP seed=${seed} Start: $(date) ==="
    log="${LogDir}/train_tcga_seed${seed}_PMA_EX_SS_AUX_AUG_DISP_ext.log"
    python -u ./train.py --dataset tcga --seed "${seed}" --PMA --rna-genes literature_1500 \
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 --image --tile-augment --attn-dispersion \
        --tile-decode-workers 8 \
        --external --group-ts "${GroupTs}" 2>&1 | tee "${log}"
    echo "=== PMA_EX_SS_AUX_AUG_DISP seed=${seed} Complete: $(date) ==="
done

echo "=== ALL PMA_EX_SS_AUX_AUG_DISP EXTERNAL RUNS COMPLETE: $(date) ==="
