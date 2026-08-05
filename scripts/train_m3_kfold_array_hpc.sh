#!/bin/bash
#SBATCH --job-name=PVT-M3-kfold-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m3_kfold_array_%a.log

# train_m1_kfold_array_hpc.sh와 동일한 이유/패턴(야간 유휴 시간대 노려서 5-fold 병렬 시도).
# race condition 픽스 반영됨(data/patch_utils.py) — 병렬로 돌려도 안전.
# M3(WSI+RNA, clinical 제외, ViT_PMA --no-clinical) — RNA가 있으니 EX+SS+AUX+AUG+DISP 전부.
#
# 2026-08-05: --rna-genes literature_1500(leaky, TCGA+CPTAC 둘 다 결합해 뽑은 both-결합
# 유전자셋) -> literature_1500_intersection으로 교체. TCGA-only/CPTAC-only 순위(각자 자기
# 코호트 라벨만으로 독립 계산)가 겹치는 유전자만 쓰는 leakage-free 대안(data/
# select_rnaseq_genes.py --intersection, 문헌 curated 163개 + 교집합 1337개 = 1500개) —
# M6/M7에서 fdr0.1(leakage-free 단일 코호트 순위)보다 external이 더 잘 나온 걸 확인한 뒤
# M3(WSI+RNA)에도 같은 방식을 적용해본다.
#
# 완료 후 집계(model_prefix에 --no-clinical의 _NOCLINICAL 태그가 들어간다는 점 주의):
#   python scripts/summarize_kfold.py --dataset tcga --seed 42 --n-folds 5 --model PMA_INT1500_SS_AUX_AUG_NOCLINICAL_DISP
#
# 제출: sbatch scripts/train_m3_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

FOLD=$SLURM_ARRAY_TASK_ID
log=".logs/train_tcga_seed42_PMA_INT1500_SS_AUX_AUG_NOCLINICAL_DISP_kfold5_fold${FOLD}.log"

echo "=== M3(PMA_INT1500_SS_AUX_AUG_NOCLINICAL_DISP) fold=${FOLD}/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --no-clinical --rna-genes literature_1500_intersection --dataset tcga --external --seed 42 \
    --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
    --fold "${FOLD}" --n-folds 5 --group-ts 0805m3_int1500_kfold5_array 2>&1 | tee "${log}"
echo "=== M3(PMA_INT1500_SS_AUX_AUG_NOCLINICAL_DISP) fold=${FOLD}/5 Complete: $(date) ==="
