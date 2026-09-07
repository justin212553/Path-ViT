#!/bin/bash
#SBATCH --job-name=PVT-porpoise-ownrna-mmf-train
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/porpoise_ownrna_mmf_train_array_%a.log

# 2026-09-06: sbatch/run_porpoise_official_paad_mmf_hpc.sh(seed=1)/_seed84_hpc.sh와 완전히
# 동일한 아키텍처/하이퍼파라미터(MMF, bilinear fusion, gate_path+gate_omic, skip, dropinput
# 0.10, nll_surv)로 genomic CSV만 sbatch/prepare_porpoise_ownrna_data_hpc.sh가 만든 own-RNA
# 버전(porpoise/datasets_csv/tcga_paad_all_clean.csv.zip, 이미 own-RNA로 덮어써짐) + own-RNA
# 전용 5-fold split(splits/5foldcv/tcga_paad_ownrna)으로 학습.
#
# main.py는 --k_start/--k_end 없이 호출하면 한 번의 실행이 5-fold 전부를 내부에서 순차 처리한다
# (기존 seed1/84 잡과 동일한 구조) — 그래서 array는 fold가 아니라 seed 축으로만 5개(84,126,
# 168,210,252, 우리 프로젝트 RNA 세트 강건성 확인용과 동일 시드 — 사용자 지시: "시드 숫자까지
# 우리랑 똑같이 해").
#
# 선행 조건: sbatch/prepare_porpoise_ownrna_data_hpc.sh 완료(own-RNA CSV/splits/CPTAC CSV 생성).
#
# 완료 후 external(CPTAC) 평가:
#   sbatch sbatch/porpoise_ownrna_mmf_eval_cptac_5seed_array_hpc.sh
#
# 제출: sbatch sbatch/porpoise_ownrna_mmf_train_5seed_array_hpc.sh

cd /pub/wonseukl/Path-ViT/porpoise

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

SEEDS=(84 126 168 210 252)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

DATA_ROOT="/pub/wonseukl/Path-ViT/porpoise/data_root_true_resnet50"
PT_FILES_DIR="/pub/wonseukl/Path-ViT/data/porpoise_style_features/tcga/pt_files"
mkdir -p "${DATA_ROOT}/tcga_paad_20x_features"
ln -sfn "${PT_FILES_DIR}" "${DATA_ROOT}/tcga_paad_20x_features/pt_files"

echo "=== PORPOISE MMF(own-RNA, true-ResNet50) seed=${SEED} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u main.py \
    --data_root_dir "${DATA_ROOT}" \
    --which_splits 5foldcv --split_dir tcga_paad_ownrna \
    --mode pathomic --model_type porpoise_mmf --bag_loss nll_surv --reg_type pathomic \
    --fusion bilinear --gate_path --gate_omic --skip --dropinput 0.10 \
    --results_dir ./results_ownrna_mmf --seed "${SEED}" --overwrite
echo "=== PORPOISE MMF(own-RNA, true-ResNet50) seed=${SEED} Complete: $(date) ==="
