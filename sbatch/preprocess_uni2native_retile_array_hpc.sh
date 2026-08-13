#!/bin/bash
#SBATCH --job-name=PVT-retile-uni2native-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-15
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/preprocess_uni2native_retile_array_%a.log

# 2026-08-12: UNI2-h 공식 스펙(256px@20x, ~0.5MPP)으로 원본 WSI(data/{tcga_paad,cptac_pda}_wsi/)를
# 우리 파이프라인으로 직접 재타일링한다 — MahmoodLab 공식 사전추출 feature(uni2official)를 쓴
# 대조실험에서 두 가지 confound가 확인됐다: (1) 공식 세트가 DX(진단용) 슬라이드만 포함해
# 환자당 슬라이드 수가 2~3장->1장으로 줄어듦(findings_backlog.md 14번 항목: 슬라이드 축소가
# external C를 0.600->0.516으로 떨어뜨린 전례가 있음), (2) 공식 coords가 level0 픽셀 좌표라
# attn_dispersion 계산 스케일이 우리 것보다 ~4000배 커서 risk_head 입력이 초기화 시점부터
# 다른 feature들을 압도함(models/spatial_features.py::attention_dispersion, dispersion_scale
# 초기값 0.2). 이 두 confound를 피하려면 우리 raw WSI 전체(DX+TS+BS 다 포함)를 우리 좌표
# 컨벤션(작은 grid index) 그대로 유지한 채 해상도만 공식 스펙으로 맞춰야 한다 —
# data/preprocess.py가 --target-mpp/--tile-size/--output-dir를 이미 지원해서 가능.
#
# --tiles-only: CNN feature 추출은 생략(GPU 필요 없음, 다음 단계인
# extract_features_uni2native_array_hpc.sh에서 UNI2-h로 별도 추출) — 타일링 자체는 GPU를 안 쓰지만
# 기존 free-gpu partition 관례를 그대로 따른다.
#
# 결과: data/patches_{tcga,cptac}_uni2native/tiles/<slide_id>/ 아래 256px JPG + slide_index_task*.csv
# (기존 data/patches_{tcga,cptac}/ 트리와 완전히 별도 — 덮어쓰기 없음, 롤백 가능)
#
# SLURM_ARRAY_TASK_ID(0~15) -> 0~7: tcga 8-shard, 8~15: cptac 8-shard.
#
# 완료 후: sbatch sbatch/extract_features_uni2native_array_hpc.sh 제출.
#
# 제출: sbatch sbatch/preprocess_uni2native_retile_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

N_SHARDS=8
IDX=$SLURM_ARRAY_TASK_ID
if [ "$IDX" -lt "$N_SHARDS" ]; then
  DATASET=tcga
  TASK_ID=$IDX
else
  DATASET=cptac
  TASK_ID=$((IDX - N_SHARDS))
fi

echo "=== retile(uni2native) dataset=${DATASET} task=${TASK_ID}/${N_SHARDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m data.preprocess --dataset "${DATASET}" \
    --output-dir "data/patches_${DATASET}_uni2native" \
    --target-mpp 0.5 --tile-size 256 --tiles-only \
    --task-id "${TASK_ID}" --num-tasks "${N_SHARDS}"
echo "=== retile(uni2native) dataset=${DATASET} task=${TASK_ID}/${N_SHARDS} Complete: $(date) ==="
