#!/bin/bash
#SBATCH --job-name=PVT-porpoise-amil
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/run_porpoise_official_paad_amil_%j.log

# 2026-09-05: PORPOISE 공식 코드(porpoise/main.py, github.com/mahmoodlab/PORPOISE 원본 —
# sota/PORPOISE에서 git-tracked 위치로 이동, torch_geometric 미사용 경로(--mode path)의
# 무조건 import 두 줄만 주석 처리, 그 외 알고리즘/모델/손실함수는 전혀 안 건드림)를
# **진짜 원본 스펙 feature**(ImageNet ResNet50, layer3 truncated, 1024차원, 256x256@20x —
# data/extract_porpoise_style_features.py, UNI2-h 공식 patch 좌표 재사용)로 처음 돌린다.
#
# 이전에(2026-09-03) 같은 코드를 우리 UNI2 feature(1536차원, 원본 아님)로 이미 한 번 완주한
# 적이 있다(.logs/sota_porpoise_paad_amil_snn.log, 5-fold val c-index 0.46~0.64, 평균 0.5대,
# 총 20epoch x 5fold 약 2.7시간) — 이번엔 feature만 진짜 원본으로 바꿔서 같은 실험을 반복.
# model_type=porpoise_amil(models/model_porpoise.py::PorpoiseAMIL, WSI 단독), mode=path
# (유전체 없음), bag_loss=nll_surv(PORPOISE 기본 — 이산시간 hazard, 4 time bin x 2 censorship
# = 8 class, Cox가 아님 — 이 손실함수 차이 자체를 그대로 유지하는 게 이번 실험의 핵심).
#
# 선행 조건: data/extract_porpoise_style_features.py 결과(data/porpoise_style_features/tcga/
# pt_files/*.pt)가 이미 있어야 함(사용자 확인: 완료됨).
#
# 완료 후: porpoise/results_true_resnet50/5foldcv/.../summary_latest.csv에 5-fold val c-index.
# .logs/sota_porpoise_paad_amil_snn.log(가짜 feature, 평균 0.5대)와 직접 비교.
#
# 제출: sbatch sbatch/run_porpoise_official_paad_amil_hpc.sh

cd /pub/wonseukl/Path-ViT/porpoise

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

# data_root_dir 아래 {study}_20x_features/pt_files/ 구조를 main.py가 요구 — 새로 추출한
# 진짜 feature(../data/porpoise_style_features/tcga/pt_files)를 그 경로로 심볼릭 링크.
DATA_ROOT="/pub/wonseukl/Path-ViT/porpoise/data_root_true_resnet50"
mkdir -p "${DATA_ROOT}/tcga_paad_20x_features"
ln -sfn /pub/wonseukl/Path-ViT/data/porpoise_style_features/tcga/pt_files "${DATA_ROOT}/tcga_paad_20x_features/pt_files"

echo "=== PORPOISE 공식 코드, 진짜 ResNet50(1024d) feature, AMIL(WSI only) Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u main.py \
    --data_root_dir "${DATA_ROOT}" \
    --which_splits 5foldcv --split_dir tcga_paad \
    --mode path --model_type porpoise_amil --bag_loss nll_surv \
    --results_dir ./results_true_resnet50 --seed 1
echo "=== PORPOISE 공식 코드, 진짜 ResNet50(1024d) feature, AMIL(WSI only) Complete: $(date) ==="
