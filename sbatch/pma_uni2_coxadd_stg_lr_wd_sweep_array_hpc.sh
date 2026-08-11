#!/bin/bash
#SBATCH --job-name=PVT-PMA-uni2-lrwd-sweep
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/pma_uni2_coxadd_stg_lr_wd_sweep_array_%a.log

# 지금까지 최선인 PMA(UNI2, INT1500, STG+R, cox_add) 레시피에서 lr/weight_decay만 스윕한다.
# tune.py(Ray Tune 기반, ViT_M1/ResNet50 전용으로 짜여있고 external eval·clinical/RNA·cox_add·
# staging 전부 지원 안 하는 레거시 코드)를 다시 손보는 대신, 이미 PMA 레시피를 전부 지원하는
# train.py를 그대로 쓰고 lr/weight_decay만 grid로 바꿔가며 도는 단순 SLURM array로 짰다 —
# 이 프로젝트에서 이미 검증된 방식(m1_pool/m2_pool/m3/pma multiseed array와 동일 패턴)이라
# Ray Tune 인프라를 새로 손볼 필요도, train.py와 별도로 유지보수할 duplicate 코드도 없다.
#
# config.py::TrainConfig 기본값(lr=1e-5, weight_decay=1e-1) 기준 위아래로 폭을 잡았다 —
# findings_backlog.md에 "WSI 스택 lr=1e-5가 왜 이렇게 낮게 잡혔는지 재검토된 적 없음"이라고
# 남아있던 항목.
#
# k-fold 없이 단일 6:2:2 split(seed84, 프로젝트 기본값) + --external로 빠르게 돈다 — 하이퍼
# 파라미터 "탐색" 단계라 val_c_index로 순위만 비교하면 되고, 최종 확정된 값만 나중에
# multi-seed k-fold로 재검증하면 된다(tune.py 원래 설계도 val_c_index 기준이었음).
#
# IDX(0~11) -> lr_idx=IDX/3, wd_idx=IDX%3 (LRS 4개 x WDS 3개 = 12조합).
#
# 완료 후: 12개 로그에서 best_val_c_index(wandb summary 또는 로그의 "checkpoint saved" 마지막
# 값)를 비교해 최댓값의 (lr, weight_decay)를 고른다:
#   grep -H "checkpoint saved" .logs/train_tcga_seed84_PMA_uni2_*LR*.log | tail -1  (파일별로)
# 또는: grep -H "best_val_c_index" .wandb/*/files/wandb-summary.json (오프라인 wandb 사용 시 생략 가능)
#
# 제출: sbatch sbatch/pma_uni2_coxadd_stg_lr_wd_sweep_array_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

LRS=(3e-6 1e-5 3e-5 1e-4)
WDS=(1e-2 1e-1 3e-1)
N_WDS=3

IDX=$SLURM_ARRAY_TASK_ID
LR_IDX=$((IDX / N_WDS))
WD_IDX=$((IDX % N_WDS))
LR=${LRS[$LR_IDX]}
WD=${WDS[$WD_IDX]}

log=".logs/train_tcga_seed84_PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_LR${LR}_WD${WD}_singlesplit.log"

echo "=== PMA(uni2,cox_add,STG+R) lr=${LR} wd=${WD} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed 84 \
    --backbone uni2 \
    --clinical-margin --clinical-staging --combine-mode cox_add \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --lr "${LR}" --weight-decay "${WD}" \
    --group-ts 0809pma_uni2_coxadd_stg_lr_wd_sweep 2>&1 | tee "${log}"
echo "=== PMA(uni2,cox_add,STG+R) lr=${LR} wd=${WD} Complete: $(date) ==="
