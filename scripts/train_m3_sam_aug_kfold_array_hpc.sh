#!/bin/bash
#SBATCH --job-name=PVT-M3-sam-aug-kfold-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m3_sam_aug_kfold_array_%a.log

# train_m3_kfold_array_hpc.sh(seed84, INT1500, AUG, no --sam)에 --sam만 추가한 버전.
#
# 배경: seed84 no-aug(internal 0.626/external 0.567) vs seed84 with-aug(internal 0.633/
# external 0.617) 비교에서, epoch별 train/val c-index 곡선을 직접 뽑아보니(사용자 요청,
# 2026-08-06) aug가 있을 때 평균적으로 val이 더 높은 지점에서 안정되고 train-val 격차도
# 작아지는 경향(0.171->0.142, 5-fold 평균)을 확인했다 — 다만 fold별 편차가 커서(fold3는
# 반대 방향) 이 작은 코호트 특유의 local-minimum 불안정성이 여전히 크다는 결론.
#
# --sam(utils/sam.py, Sharpness-Aware Minimization)은 정확히 이 문제(초기화만 바꿔도
# external이 크게 흔들리는 local-minimum 로또)를 겨냥해 이미 구현돼 있었지만 이 프로젝트에서
# 한 번도 실사용된 적이 없다 — flat minimum을 명시적으로 찾도록 강제해 fold/seed 간 불안정성
# 자체를 줄일 수 있는지 확인한다. AUG(patch-keep-frac 0.8 SS도 포함)와 함께 켠다 — 이미 aug가
# val을 안정시키는 효과가 일부 확인된 상태에서, SAM이 추가로 fold 간 분산을 줄이는지 본다.
#
# 2026-08-06: model_prefix에 _SAM{rho} 접미사가 자동으로 붙도록 train.py를 고쳤다(전엔 --sam
# 유무가 model_prefix/checkpoint/kfold_preds 파일명에 전혀 반영되지 않아, 이 스크립트를 그냥
# 돌리면 이미 받아둔 no-SAM seed84 aug 결과를 덮어썼을 뻔했다 — _AUG/_NOSPATIAL을 빠뜨렸던
# 것과 같은 사고 클래스).
#
# 배치당 forward+backward가 2번이라 학습 시간이 대략 2배 — no-SAM 버전이 fold당 실측
# ~3.2~3.7시간이었으니 SAM은 fold당 ~7시간 안팎으로 예상, --time=24h 안에 충분히 들어온다.
#
# 완료 후 집계(model_prefix에 --no-clinical의 _NOCLINICAL, --sam의 _SAM0.05 태그가 들어간다는 점 주의):
#   python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model PMA_INT1500_SS_AUX_AUG_NOCLINICAL_DISP_SAM0.05
#
# 제출: sbatch scripts/train_m3_sam_aug_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

FOLD=$SLURM_ARRAY_TASK_ID
log=".logs/train_tcga_seed84_PMA_INT1500_SS_AUX_AUG_NOCLINICAL_DISP_SAM0.05_kfold5_fold${FOLD}.log"

echo "=== M3(PMA_INT1500_SS_AUX_AUG_NOCLINICAL_DISP_SAM0.05) fold=${FOLD}/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --no-clinical --rna-genes literature_1500_intersection --dataset tcga --external --seed 84 \
    --tile-augment --image --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --sam --sam-rho 0.05 \
    --tile-decode-workers 8 --cache-val-tiles --cache-external-tiles \
    --fold "${FOLD}" --n-folds 5 --group-ts 0806m3_int1500_sam_aug_kfold5_array_seed84 2>&1 | tee "${log}"
echo "=== M3(PMA_INT1500_SS_AUX_AUG_NOCLINICAL_DISP_SAM0.05) fold=${FOLD}/5 Complete: $(date) ==="
