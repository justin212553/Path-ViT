#!/bin/bash
#SBATCH --job-name=PVT-HDPPC-lrsweep
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/hdp_pretrain_cluster_lr_sweep_seed84_fold0_%a.log

# HDP_PRETRAIN_CLUSTER(train_hdp_pretrain_cluster.py) 2seed x 5fold 학습 로그(paper/hdp/*.log)를
# 사용자가 직접 확인 — loss가 후반 epoch로 갈수록 단조 감소가 아니라 계속 진동(예: seed84/fold0
# epoch12~21에서 0.57→0.54→0.67→0.52→0.55→0.60→0.49→0.64→0.61→0.66, seed126/fold0은 epoch30
# 이후 거의 노이즈 수준으로 진동)하는 걸 발견했다. 이 lr=1e-3(M6/M7/HDP에서 이미 검증된 값이지만
# 그건 RNAEncoder+선형 cox_add 얘기였음)이 새로 초기화된 GrowthPatternCNN+MaturityMLP에는 너무
# 높을 수 있다는 가설(사용자) — train.py(WSI ViT+ABMIL 포함 스택)의 원래 lr=1e-5까지 포함해서
# 같은 seed(84)/fold(0)로 lr만 바꿔가며 스윕한다.
#
# array 관례: SLURM_ARRAY_TASK_ID(0~4) -> LRS 배열 인덱스.
#   0: 1e-3(현재 기본값, 비교 기준)
#   1: 3e-4
#   2: 1e-4
#   3: 3e-5
#   4: 1e-5(train.py/원래 PMA 스택 값)
#
# 완료 후: 5개 로그(hdp_pretrain_cluster_lr_sweep_seed84_fold0_{0..4}.log)에서 loss/val_c_index
# 곡선이 lr별로 얼마나 매끄러운지, best epoch의 val_c_index/internal test_c_index가 어떻게
# 달라지는지 비교. 안정화되는 lr을 찾으면 그 값으로 2seed x 5fold 전체를 재실행
# (hdp_pretrain_cluster_multiseed_kfold_array_hpc.sh에 --lr 인자 추가해서).
#
# 제출: sbatch sbatch/hdp_pretrain_cluster_lr_sweep_seed84_fold0_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

LRS=(1e-3 3e-4 1e-4 3e-5 1e-5)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

log=".logs/train_tcga_seed84_HDP_PRETRAIN_CLUSTER_INT1500_STG_R_GROWTH8_LR${LR}_fold0.log"

echo "=== HDP_PRETRAIN_CLUSTER lr=${LR} seed=84 fold=0/5 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u train_hdp_pretrain_cluster.py --dataset tcga --external --seed 84 \
    --fold 0 --n-folds 5 \
    --lr "${LR}" \
    --epochs 100 --patience 20 \
    --group-ts 0901hdp_pretrain_cluster_lr_sweep 2>&1 | tee "${log}"
echo "=== HDP_PRETRAIN_CLUSTER lr=${LR} seed=84 fold=0/5 Complete: $(date) ==="
