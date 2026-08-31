#!/bin/bash
#SBATCH --job-name=PVT-PORPOISE-ablation
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_uni2_stg_r_ablation_seed84_fold0_%a.log

# ViT_PORPOISE(--PORPOISE) seed84/fold0 파일럿(internal C=0.7063, sbatch/
# porpoise_uni2_stg_r_pilot_seed84_fold0_hpc.sh)이 --attn-dispersion/--rna-aux-weight를 둘 다
# 켠 채로 나온 결과라, 이 두 보조 장치가 실제로 기여하는지 아니면 그냥 딸려온 것뿐인지
# (이전에 M4/PMA에서 "dispersion·rna-aux 둘 다 큰 영향 없다"는 결론이 났던 전례)를 PORPOISE
# 에서도 확인한다. array 3개로 dispersion만 빼기 / aux만 빼기 / 둘 다 빼기를 한 번에 돌린다.
# (array index 0 = 파일럿과 동일, 즉 "둘 다 켠" baseline 재확인용은 이미 파일럿 로그에 있으므로
# 여기선 만들지 않음 — 0/1/2 = dispersion만 뺌 / aux만 뺌 / 둘 다 뺌)
#
# 완료 후 각 로그의 internal test_c_index를 파일럿의 0.7063과 나란히 비교.
#
# 제출: sbatch sbatch/porpoise_uni2_stg_r_ablation_seed84_fold0_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

IDX=$SLURM_ARRAY_TASK_ID

case $IDX in
  0)
    LABEL="no_dispersion"
    EXTRA_FLAGS="--rna-aux-weight 1.0"
    ;;
  1)
    LABEL="no_aux"
    EXTRA_FLAGS="--attn-dispersion"
    ;;
  2)
    LABEL="no_dispersion_no_aux"
    EXTRA_FLAGS=""
    ;;
esac

echo "=== PORPOISE ablation(${LABEL}) seed=84 fold=0/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PORPOISE --rna-genes literature_1500_intersection --dataset tcga --seed 84 \
    --backbone uni2 \
    --clinical-margin --clinical-staging \
    --patch-keep-frac 0.8 ${EXTRA_FLAGS} \
    --fold 0 --n-folds 5 --group-ts "0831porpoise_uni2_stg_r_ablation_${LABEL}"
echo "=== PORPOISE ablation(${LABEL}) seed=84 fold=0/5 Complete: $(date) ==="
