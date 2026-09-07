#!/bin/bash
#SBATCH --job-name=PVT-porpoise-mmf-seed84
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/run_porpoise_official_paad_mmf_seed84_%j.log

# 2026-09-06: run_porpoise_official_paad_mmf_hpc.sh(--seed 1, main.py 기본값)와 완전히 동일한
# 설정으로 --seed 84만 다르게 돌린다 — PORPOISE 논문이 시드를 전혀 공개하지 않고(Methods 원문
# 확인: "5-fold CV를 5번 반복"만 언급, 여러 시드 반복/평균 언급 없음) seed=1 결과(pooled
# C=0.5958)가 논문 0.653보다 낮게 나온 게 "재현 실패"가 아니라 "이 코호트 규모(147명)에서
# 자연스러운 시드 변동성"일 가능성을 검증하기 위함(사용자 결정 — 재현을 더 밀어붙이기보다
# "PORPOISE는 재현성 검증이 안 된 단일 시드 결과, 우리는 다중 시드로 분산까지 정량화했다"는
# 이 논문 전체의 방법론적 강점으로 삼기로 함).
#
# 84는 이 프로젝트가 자체 모델 전체에서 표준으로 쓰는 2seed(84,126) 관례의 첫 번째 시드를
# 그대로 가져온 것 — PORPOISE 재현도 이 프로젝트와 동일하게 "5fold x 2seed" 구색을 맞춰
# apple-to-apple 비교가 되게 한다(사용자 지시).
#
# results_dir은 그대로(./results_true_resnet50_mmf) 둬도 된다 — main.py가 경로 끝에
# "_s{seed}"를 자동으로 붙이므로(porpoise/main.py:238) seed=1 결과와 겹치지 않고 별도
# 하위 폴더에 쌓인다. --overwrite도 불필요(새 경로라 덮어쓸 게 없음)하지만 재실행 안전성을
# 위해 그대로 유지.
#
# 완료 후, seed 1+84 합쳐서 우리 프로젝트 관례(pooled+앙상블)로 재계산:
#   python scripts/pool_porpoise_official_kfold.py --results-dir porpoise/results_true_resnet50_mmf \
#       --seeds 1,84 --bootstrap 2000
#
# 제출: sbatch sbatch/run_porpoise_official_paad_mmf_seed84_hpc.sh

cd /pub/wonseukl/Path-ViT/porpoise

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

DATA_ROOT="/pub/wonseukl/Path-ViT/porpoise/data_root_true_resnet50"
PT_FILES_DIR="/pub/wonseukl/Path-ViT/data/porpoise_style_features/tcga/pt_files"
mkdir -p "${DATA_ROOT}/tcga_paad_20x_features"
ln -sfn "${PT_FILES_DIR}" "${DATA_ROOT}/tcga_paad_20x_features/pt_files"

echo "=== 슬라이드 존재 여부로 CSV 필터링: $(date) ==="
python -u filter_available_slides.py --pt-files-dir "${PT_FILES_DIR}"

echo "=== PORPOISE 공식 코드, 진짜 ResNet50(1024d) feature, MMF, seed=84 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u main.py \
    --data_root_dir "${DATA_ROOT}" \
    --which_splits 5foldcv --split_dir tcga_paad \
    --mode pathomic --model_type porpoise_mmf --bag_loss nll_surv --reg_type pathomic \
    --fusion bilinear --gate_path --gate_omic --skip --dropinput 0.10 \
    --results_dir ./results_true_resnet50_mmf --seed 84 --overwrite
echo "=== PORPOISE 공식 코드, 진짜 ResNet50(1024d) feature, MMF, seed=84 Complete: $(date) ==="
