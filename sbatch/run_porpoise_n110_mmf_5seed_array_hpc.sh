#!/bin/bash
#SBATCH --job-name=PVT-porpoise-n110-train
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-24
#SBATCH --requeue
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/run_porpoise_n110_mmf_5seed_array_%a.log

# 2026-10-10: PORPOISE 공식 아키텍처(MMF)를 새 프로토콜 C안으로 재학습 — 우리 M1~M7과 같은 코호트
# (TCGA 110), 같은 seed별 fold 배정, 같은 RNA 입력(pdac_consistency_1500 raw log2, PORPOISE 자체
# fold-train scaler = fold-safe). WSI는 PORPOISE 공식 스펙 ResNet50 feature. 레시피(bilinear fusion,
# gate, skip, dropinput 0.10, nll_surv)는 공식 그대로.
# 2026-10-10 수정: 한 잡이 5 fold를 순차 학습하다 free-gpu preemption으로 중단됨 → task 하나가 fold 하나
# (--k_start f --k_end f+1). main.py가 fold마다 seed_torch(seed)를 다시 호출하므로 결과는 순차 실행과 같다.
# array index = seed_idx*5 + fold(eval 스크립트와 같은 규칙 → eval은 --dependency=aftercorr로 1:1 대기).
# --overwrite 필수: main.py는 summary_latest.csv가 있으면 실행을 건너뛰는데, 같은 seed의 다른 fold task가
# 먼저 끝나며 그 파일을 쓰기 때문.
#
# 선행: python -m scripts.prepare_porpoise_paad_n110 (repo 루트에서)
# 다음: sbatch --dependency=aftercorr:<이 잡 ID> sbatch/eval_porpoise_n110_mmf_external_cptac_5seed_hpc.sh

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

echo "=== PORPOISE N110 MMF seed=${SEED} fold=${FOLD} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u main.py \
    --data_root_dir "${DATA_ROOT}" \
    --which_splits 5foldcv_n110 --split_dir "tcga_paad_seed${SEED}" \
    --mode pathomic --model_type porpoise_mmf --bag_loss nll_surv --reg_type pathomic \
    --fusion bilinear --gate_path --gate_omic --skip --dropinput 0.10 \
    --k_start "${FOLD}" --k_end "$((FOLD + 1))" \
    --results_dir ./results_n110_mmf --seed "${SEED}" --overwrite
echo "=== PORPOISE N110 MMF seed=${SEED} fold=${FOLD} Complete: $(date) ==="
