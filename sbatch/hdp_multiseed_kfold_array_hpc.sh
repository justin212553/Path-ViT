#!/bin/bash
#SBATCH --job-name=PVT-HDP-multiseed-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/hdp_multiseed_kfold_array_%a.log

# HDP(Human Doctor Prognosis, models/hdp.py, train_light.py --HDP) — M7(age/sex+RNA-seq)에
# WSI 유래 저차원 구조화 feature 하나를 추가한 첫 버전. 2026-09-01 WSI branch 재설계 논의의
# 결론: attention/MIL로 patch 중요도를 152개 생존 라벨만으로 처음부터 발견시키는 접근은
# 이번 세션 내내(MCAT/PORPOISE/sharpening) 실패했고, "조직 유형에 사람이 이름 붙이는" 대안도
# TCGA/CPTAC 어디에도(clinical.tsv, pathology_detail.tsv 전부 확인) PAAD용 PNI/TIL/종양비율
# 라벨이 없어 불가능했다(사용자 지적: "나도 의사가 아니라 이걸 봐도 모른다"). 그래서 군집에
# 의미를 붙이지 않고, UNI2-h(공식 스펙, 20x/~0.5um px) frozen feature를 라벨 없이 k-means로
# 군집화한 뒤(data/fit_clusters_uni2native.py, K=10) 환자별 군집 비율(K차원 히스토그램,
# data/compute_cluster_histograms_uni2native.py) 자체를 clinical/RNA와 동일한 cox_add 방식으로
# 넣는다 — 어느 군집 비율이 hazard와 상관있는지는 생존 라벨 자체가 학습 중 결정한다.
#
# WSI patch forward(CNN/ViT/ABMIL) 자체가 전혀 없다(히스토그램은 이미 사전 계산됨) — train.py가
# 아니라 훨씬 가벼운 train_light.py(M5/M6/M6X/M7과 같은 하네스) 사용. M7 레퍼런스 레시피
# (epochs=100, patience=20, lr=1e-3)를 그대로 따른다.
#
# ⚠️ 실행 전 필수: 이 job은 로컬에서 계산된 두 파일이 HPC에도 있어야 한다(로컬 전용, git
# 미포함 — data/uni2h_official_features/ 45GB는 필요 없고 이 두 CSV만 있으면 됨):
#   data/cluster_hist_uni2native_tcga.csv
#   data/cluster_hist_uni2native_cptac.csv
# (data/cluster_centroids_uni2native.pt는 학습 자체엔 불필요 — 히스토그램 계산에만 쓰였음)
#
# array 관례: SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5 (2seed x 5fold).
# 논문 관례대로 seed42는 제외(WSI 계열 모델은 seed42에서 유독 성능이 안 나옴, paper/
# results_table_pma_family_3seed_kfold_ci.md) — seed 84/126만 사용.
#
# 완료 후(10개 fold 로그 전부 확인, .logs/kfold_preds/에 CSV 10개 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
#       --model HDP_INT1500_STG_R --seeds 84,126 --n-folds 5 --bootstrap 2000
#
# 그다음 external 평가는 sbatch/hdp_multiseed_external_eval_hpc.sh(재학습 없이
# eval-external-ckpt로 checkpoint 재사용, 이 10개 학습이 전부 끝난 뒤 제출).
#
# 제출: sbatch sbatch/hdp_multiseed_kfold_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5
IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX / N_FOLDS))
FOLD=$((IDX % N_FOLDS))
SEED=${SEEDS[$SEED_IDX]}

log=".logs/train_tcga_seed${SEED}_HDP_INT1500_STG_R_kfold5_fold${FOLD}.log"

echo "=== HDP seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u train_light.py --HDP --dataset tcga --external --seed "${SEED}" \
    --rna-genes literature_1500_intersection \
    --clinical-margin --clinical-staging \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --epochs 100 --patience 20 \
    --group-ts 0901hdp_multiseed_kfold5_array 2>&1 | tee "${log}"
echo "=== HDP seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
