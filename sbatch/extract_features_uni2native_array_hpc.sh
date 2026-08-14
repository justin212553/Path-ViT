#!/bin/bash
#SBATCH --job-name=PVT-extract-uni2native-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/extract_uni2native_array_%a.log

# preprocess_uni2native_retile_array_hpc.sh(256px@0.5MPP 재타일링, data/patches_{tcga,cptac}_uni2native/)
# 완료 후, 그 타일에 UNI2-h feature를 추출한다. --backbone uni2native가
# UNI2_NATIVE_PATCH_TRANSFORM(리사이즈 없음, 이미 256px)을 쓴다(utils/extract_features.py
# BACKBONE_REGISTRY 참조) — 기존 "uni2"(1024px 원본을 512로 리사이즈)와 다른 항목.
#
# [2026-08-13] 최초 버전(--array=0-1, 데이터셋당 1개 job이 전체를 순차 처리)은 슬라이드당 타일
# 수가 최대 ~58배로 폭증해(uni2official 대조실험에서 실측) 단일 job 기준 77시간+ 예상으로 너무
# 느렸다 — utils/extract_features.py에 --task-id/--num-tasks 샤딩을 추가(data/preprocess.py와
# 동일 관례)했다. TCGA는 CPTAC보다 훨씬 느려서(native가 대부분 40x라 다운샘플 여유가 적음)
# 10-shard, CPTAC은 36시간 추정이라 5-shard로 비대칭 분할한다. 이미 처리된 슬라이드는 skip하므로
# 재실행도 안전하다.
#
# 결과: data/patches_{tcga,cptac}_uni2native/tiles/<slide_id>/features_uni2.pt
#   (별도 디렉토리 트리라 out_filename은 그냥 기본 features_uni2.pt — 기존 산출물과 안 겹침)
#
# SLURM_ARRAY_TASK_ID(0~14) -> 0~9: tcga 10-shard, 10~14: cptac 5-shard.
#
# 완료 후: scripts/reconcile_uni2native_features.py로 기존 patches 트리에
#   features_uni2native.pt/coords_uni2native.pt로 복사(로컬 다운로드는 이 작은 결과물만).
#
# 제출: sbatch sbatch/extract_features_uni2native_array_hpc.sh
# (재타일링이 끝난 데이터셋만 의미 있음 — tcga가 아직 안 끝났으면 cptac 몫(10~14)만 먼저
#  제출하고 싶을 때는 --array=10-14로 좁혀서 별도 제출해도 된다)

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N_SHARDS_TCGA=10
N_SHARDS_CPTAC=5
IDX=$SLURM_ARRAY_TASK_ID
if [ "$IDX" -lt "$N_SHARDS_TCGA" ]; then
  DATASET=tcga
  N_SHARDS=$N_SHARDS_TCGA
  TASK_ID=$IDX
else
  DATASET=cptac
  N_SHARDS=$N_SHARDS_CPTAC
  TASK_ID=$((IDX - N_SHARDS_TCGA))
fi

echo "=== extract_features(uni2native) dataset=${DATASET} task=${TASK_ID}/${N_SHARDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m utils.extract_features --dataset "${DATASET}" --backbone uni2native \
    --patches-root "data/patches_${DATASET}_uni2native" \
    --task-id "${TASK_ID}" --num-tasks "${N_SHARDS}"
echo "=== extract_features(uni2native) dataset=${DATASET} task=${TASK_ID}/${N_SHARDS} Complete: $(date) ==="
