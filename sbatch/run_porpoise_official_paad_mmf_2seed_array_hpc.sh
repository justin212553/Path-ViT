#!/bin/bash
#SBATCH --job-name=PVT-porpoise-mmf-2seed
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/run_porpoise_official_paad_mmf_2seed_array_%a.log

# 2026-09-12: run_porpoise_official_paad_mmf_hpc.sh(seed=1)와 run_porpoise_official_paad_mmf_
# seed84_hpc.sh(seed=84, 실제로는 제출된 적 없었음)를 seed 84/126 2개짜리 array 하나로 합친다
# (사용자 지시) — 이 프로젝트의 다른 모든 모델이 쓰는 표준 2seed(84,126) 관례에 맞추기 위해
# seed=1은 더 이상 쓰지 않는다.
#
# porpoise/main.py는 --which_splits 5foldcv --split_dir tcga_paad로 호출하면 그 안에서 5개
# fold를 전부 순회한다(main.py:47-92, --k 기본값 5) — 이 프로젝트 자체 파이프라인(train.py)과
# 달리 fold별로 따로 제출할 필요가 없다. 그래서 array는 fold가 아니라 seed 2개만 돈다.
#
# 모델 설정은 run_porpoise_official_paad_mmf_hpc.sh와 완전히 동일 — fusion=bilinear,
# gate_path/gate_omic/skip 전부 명시(안 주면 main.py CLI 기본값이 PorpoiseMMF 클래스 기본값을
# 덮어써서 genomics 브랜치가 없는 반쪽 모델이 됨, 그 스크립트의 코멘트 참조), dropinput=0.10.
# results_dir은 그대로 두면 main.py가 "_s{seed}"를 자동으로 붙이므로(main.py:238) 두 시드가
# 서로 다른 하위 폴더에 쌓이고 덮어쓰지 않는다.
#
# 완료 후(두 seed 다 s_0~s_4_checkpoint.pt 5개씩 있는지 확인):
#   python scripts/pool_porpoise_official_kfold.py --results-dir porpoise/results_true_resnet50_mmf \
#       --seeds 84,126 --bootstrap 2000
#
# external(CPTAC) 평가는 eval_porpoise_official_paad_mmf_external_cptac_hpc.sh(seed 84/126로
# 갱신됨)를 이 array 완료 후 제출.
#
# 제출: sbatch sbatch/run_porpoise_official_paad_mmf_2seed_array_hpc.sh

cd /pub/wonseukl/Path-ViT/porpoise

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

DATA_ROOT="/pub/wonseukl/Path-ViT/porpoise/data_root_true_resnet50"
PT_FILES_DIR="/pub/wonseukl/Path-ViT/data/porpoise_style_features/tcga/pt_files"
mkdir -p "${DATA_ROOT}/tcga_paad_20x_features"
ln -sfn "${PT_FILES_DIR}" "${DATA_ROOT}/tcga_paad_20x_features/pt_files"

SEEDS=(84 126)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== 슬라이드 존재 여부로 CSV 필터링: $(date) ==="
python -u filter_available_slides.py --pt-files-dir "${PT_FILES_DIR}"

echo "=== PORPOISE 공식 코드, 진짜 ResNet50(1024d) feature, MMF(WSI+genomics, gated bilinear fusion), seed=${SEED} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u main.py \
    --data_root_dir "${DATA_ROOT}" \
    --which_splits 5foldcv --split_dir tcga_paad \
    --mode pathomic --model_type porpoise_mmf --bag_loss nll_surv --reg_type pathomic \
    --fusion bilinear --gate_path --gate_omic --skip --dropinput 0.10 \
    --results_dir ./results_true_resnet50_mmf --seed "${SEED}" --overwrite
echo "=== PORPOISE 공식 코드, 진짜 ResNet50(1024d) feature, MMF, seed=${SEED} Complete: $(date) ==="
