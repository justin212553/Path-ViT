#!/bin/bash
# HPC가 점검 중이라 sbatch/porpoise_attn_temperature_pilot_seed84_fold0_hpc.sh +
# porpoise_entropy_reg_pilot_seed84_fold0_hpc.sh + porpoise_sharpening_with_aux_pilot_
# seed84_fold0_hpc.sh(총 10개 job)를 로컬에서 순차 실행하는 버전 — 세 sbatch 스크립트와
# 정확히 동일한 레시피/플래그.
#
# 사용법: PathViT-ray conda env에서
#   bash scripts/run_porpoise_sharpening_pilots_local.sh
set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

_run() {
  local tag="$1"; shift
  local log=".logs/porpoise_${tag}_seed84_fold0.log"
  echo "=== ${tag} seed=84 fold=0/5 Start: $(date) ==="
  python -u ./train.py --PORPOISE "$@" \
      --rna-genes literature_1500_intersection --dataset tcga --seed 84 \
      --backbone uni2 \
      --clinical-margin --clinical-staging \
      --patch-keep-frac 0.8 --attn-dispersion \
      --fold 0 --n-folds 5 --group-ts "0901porpoise_${tag}" 2>&1 | tee "${log}"
  echo "=== ${tag} seed=84 fold=0/5 Complete: $(date) ==="
}

echo "########## [1/3] attn-temperature 스윕(aux=0) ##########"
_run "temp0.5" --porpoise-attn-temperature 0.5
_run "temp0.2" --porpoise-attn-temperature 0.2
_run "temp0.1" --porpoise-attn-temperature 0.1

echo "########## [2/3] entropy-reg-weight 스윕(aux=0) ##########"
_run "entreg0.05" --entropy-reg-weight 0.05
_run "entreg0.1"  --entropy-reg-weight 0.1
_run "entreg0.3"  --entropy-reg-weight 0.3

echo "########## [3/3] sharpening + rna-aux-weight=1.0 ##########"
_run "temp0.2_aux1.0"   --porpoise-attn-temperature 0.2 --rna-aux-weight 1.0
_run "temp0.1_aux1.0"   --porpoise-attn-temperature 0.1 --rna-aux-weight 1.0
_run "entreg0.1_aux1.0" --entropy-reg-weight 0.1 --rna-aux-weight 1.0
_run "entreg0.3_aux1.0" --entropy-reg-weight 0.3 --rna-aux-weight 1.0

echo "=== 전부 완료: $(date) ==="
