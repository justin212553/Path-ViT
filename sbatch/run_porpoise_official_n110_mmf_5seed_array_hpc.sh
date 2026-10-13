#!/bin/bash
#SBATCH --job-name=PVT-porpoise-off110-train
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-24
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/run_porpoise_official_n110_mmf_5seed_array_%a.log

# 2026-10-11: 시스템 수준 비교 — PORPOISE 공식 입력(datasets_csv_mutsig: RNA 1553 + CNV 82 + mutation 14)을
# 그대로 쓰고 프로토콜만 우리와 같게(같은 seed별 fold 배정 5x5, 공식 CSV에 있는 우리 코호트 103명).
# 코드·레시피는 공식 그대로. task 하나 = fold 하나(preemption 대응, run_porpoise_n110_*과 같은 구조).
# array index = seed_idx*5 + fold. --overwrite 필요 이유는 run_porpoise_n110_mmf_5seed_array_hpc.sh 참조.
#
# 선행(로그인 노드에서 1회, 25개 task가 같은 CSV를 동시에 고쳐 쓰지 않도록 잡 밖에서):
#   python -m scripts.prepare_porpoise_paad_n110 --official-geneset          (repo 루트)
#   cd porpoise && python filter_available_slides.py \
#       --pt-files-dir /pub/wonseukl/Path-ViT/data/porpoise_style_features/tcga/pt_files \
#       --csv-path datasets_csv_mutsig/tcga_paad_all_clean.csv.zip          (feature 없는 슬라이드 행 제거, .orig 1회 백업)
# 다음: sbatch --dependency=aftercorr:<이 잡 ID> sbatch/eval_porpoise_official_n110_mmf_external_cptac_5seed_hpc.sh

cd /pub/wonseukl/Path-ViT/porpoise
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

SEEDS=(84 126 42 168 210)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((IDX / N_FOLDS))]}
FOLD=$((IDX % N_FOLDS))
DATA_ROOT="/pub/wonseukl/Path-ViT/porpoise/data_root_true_resnet50"
PT_FILES_DIR="/pub/wonseukl/Path-ViT/data/porpoise_style_features/tcga/pt_files"
mkdir -p "${DATA_ROOT}/tcga_paad_20x_features"
ln -sfn "${PT_FILES_DIR}" "${DATA_ROOT}/tcga_paad_20x_features/pt_files"

echo "=== PORPOISE OFFICIAL-GENESET N110 MMF seed=${SEED} fold=${FOLD} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u main.py \
    --data_root_dir "${DATA_ROOT}" \
    --which_splits 5foldcv_n110official${QC:+qc} --split_dir "tcga_paad_seed${SEED}" \
    --mode pathomic --model_type porpoise_mmf --bag_loss nll_surv --reg_type pathomic \
    --apply_mutsig \
    --fusion bilinear --gate_path --gate_omic --skip --dropinput 0.10 \
    --k_start "${FOLD}" --k_end "$((FOLD + 1))" \
    --results_dir ./results_official_n110_mmf${QC:+_qc} --seed "${SEED}" --overwrite
echo "=== PORPOISE OFFICIAL-GENESET N110 MMF seed=${SEED} fold=${FOLD} Complete: $(date) ==="
