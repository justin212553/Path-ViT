#!/bin/bash
#SBATCH --job-name=PVT-M1-uni2native-pilot
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/m1_uni2native_seed42_fold0_pilot_%a.log

# M1(WSI 단독)의 internal c-index가 0.5 밑(aux 없이 0.4541)이라 --stage-aux-weight로
# 0.5139까지 끌어올렸는데, stage-aux는 M1이 "WSI만 입력"이라는 모달리티 사다리 전제를
# 깨는 방식(clinical staging 데이터가 학습중 gradient로 새어들어감)이라는 문제가 있다.
# PMA_uni2native 3seed x 5fold 전체 결과(internal 0.6359->0.6313, external 0.6337->0.6238,
# 둘 다 노이즈 범위 내 flat)를 보면 전체 모델 스케일에서는 재타일링(256px@0.5MPP, DX-only/
# coords 스케일 confound 없는 버전)이 큰 차이를 안 만들었지만, M1은 다른 모달리티로 보완이
# 안 되는 유일한 모델이라 타일 품질 영향을 더 크게 받을 수 있다 — 그 가설을 저비용
# fold0/seed42 파일럿 2개로 확인한다.
#
# array index 0: M1 aux-free(순수 WSI-only, 모달리티 순수성 유지) + uni2native
# array index 1: M1 + stage-aux-weight(현재 0.5 넘기는 유일한 방법) + uni2native (같이 쓰면 더 오르는지)
#
# 사전조건: uni2native feature가 이미 HPC에 존재해야 함(features_uni2native.pt/
# coords_uni2native.pt, scripts/reconcile_uni2native_features.py 산출물 — PMA_uni2native
# 실험 때 이미 만들어둔 것 재사용, 추가 추출 불필요).
#
# 완료 후 test_c_index(로그 마지막 "Test C-index" 줄) 두 값을 aux-free/aux-included
# uni2(기존) 값과 비교:
#   M1 aux-free  uni2(기존)=0.4541(internal 5fold pooled) vs uni2native fold0 단일값
#   M1 stage-aux uni2(기존)=0.5139(internal 5fold pooled) vs uni2native fold0 단일값
#
# 제출: sbatch sbatch/m1_uni2native_seed42_fold0_pilot_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=42
FOLD=0
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID

if [ "$IDX" -eq 0 ]; then
  TAG="M1_uni2native_DISP_NOVIT"
  log=".logs/train_tcga_seed${SEED}_${TAG}_fold${FOLD}_pilot.log"
  echo "=== M1(uni2native,skip-patch-vit,DISP, aux-free) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
  python -u ./train.py --M1 --dataset tcga --external --seed "${SEED}" \
      --backbone uni2native \
      --attn-dispersion \
      --skip-patch-vit \
      --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m1_uni2native_seed42_fold0_pilot 2>&1 | tee "${log}"
  echo "=== M1(uni2native,skip-patch-vit,DISP, aux-free) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
else
  TAG="M1_uni2native_AUX2_DISP_NOVIT"
  log=".logs/train_tcga_seed${SEED}_${TAG}_fold${FOLD}_pilot.log"
  echo "=== M1(uni2native,skip-patch-vit,DISP,stage-aux) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
  python -u ./train.py --M1 --skip-patch-vit --attn-dispersion --stage-aux-weight 1.0 \
      --dataset tcga --external --seed "${SEED}" \
      --backbone uni2native \
      --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m1_uni2native_seed42_fold0_pilot 2>&1 | tee "${log}"
  echo "=== M1(uni2native,skip-patch-vit,DISP,stage-aux) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
fi
