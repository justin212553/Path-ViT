#!/bin/bash
#SBATCH --job-name=PVT-attn-heatmaps
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/generate_attention_heatmaps.log

# scripts/generate_attention_heatmaps.py — M4/PMA의 ABMIL patch attention + co-attention
# 4-component 가중치를 실제 WSI 위에 시각화(2026-08-31, "우선 시각적 확인 먼저 해보자" 사용자
# 지시). baseline(기존 3seed HPC 배치 체크포인트)과 no_coattn/no_abmil/no_nystrom(단일
# seed84/fold0, m4_pma_no_{coattn,abmil,nystrom}_kfold_hpc.sh) 4개를 나란히 비교한다 —
# findings_backlog.md의 "attention entropy가 0.999~1.000으로 붕괴" 정량 진단을 눈으로 재확인.
#
# 필요 조건: m4_pma_no_coattn/no_abmil/no_nystrom_kfold_hpc.sh 세 개가 먼저 완료돼 있어야
# 4-variant 비교가 가능하다(체크포인트 없으면 FileNotFoundError로 바로 실패). baseline만 먼저
# 보고 싶으면 --variant baseline 인자로 좁혀서 실행(아래 python 커맨드 수정).
#
# ⚠️ 로컬에 uni2native WSI 타일/체크포인트가 전혀 없어(전부 HPC 전용) 이 스크립트는
# 로컬에서 스모크테스트를 못 했다 — 이 세션의 다른 스크립트들과 달리 처음 실행이 곧 첫
# 실전 검증이다. 실패하면 에러 메시지 그대로 알려줄 것.
#
# 결과: .logs/attention_heatmaps/<case_id>_<variant들>.png (환자당 1장, variant x slide 격자)
#
# 제출: sbatch sbatch/generate_attention_heatmaps_hpc.sh

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== generate_attention_heatmaps Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.generate_attention_heatmaps
echo "=== generate_attention_heatmaps Complete: $(date) ==="
