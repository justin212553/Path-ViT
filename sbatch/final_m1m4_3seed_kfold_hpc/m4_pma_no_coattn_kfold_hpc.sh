#!/bin/bash
#SBATCH --job-name=PVT-M4PMA-NOCOATTN-1run
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/paper/.hpc/m4_pma_no_coattn_kfold.log

# M4/PMA ablation 3종 중 co-attention 담당 — WSI가 성능에 안 먹히는 원인이 Nystrom self-attn/
# ABMIL(MultiComponentPooling attn view)/co-attention(RNA-query) 중 무엇인지 하나씩 분리해서
# 본다(2026-08-31, 사용자 지시: "Nystrom self attention에서 문제인 건지 ABMIL이 문제인건지,
# 아니면 co-attention이 문제인 건지 부터 확인해야겠어"). m4_pma_3seed_kfold_array_hpc.sh(최종
# 채택 레시피)에서 --no-coattn(신규 플래그, models/vit_pma.py::ViT_PMA use_coattn 참조)만
# 추가 — 나머지(backbone/clinical/rna-aux/dispersion/combine-mode)는 전부 동일.
#
# --no-coattn: component_coattn(RNA가 4개 pooling 관점 중 뭘 볼지 고르는 co-attention)을 아예
# 안 만들고, 대신 4개 관점을 단순 평균한다 — "RNA-query 기반 관점 선택"이 실제로 도움되는지,
# 아니면 그냥 다 균등하게 봐도 차이가 없는지 검증.
#
# 2026-08-31: 정식 통계 검정(2seed x 5fold + paired bootstrap)이 아니라 "일단 방향성부터
# 빠르게 보자"는 목적이라 단일 seed x 단일 fold로 축소(사용자 지시: "kfold 말고 그냥 원시드
# 원폴드로 가지"). 방향이 흥미로우면 그때 2seed x 5fold로 정식 검정하면 됨.
#
# 태그: PMA_uni2native_INT1500_SS_AUX_STG_R_NOCOATTN_DISP_COX_ADD (model_prefix 부착 순서상
# _NOCOATTN이 _R과 _DISP 사이에 낌 — train.py 확인함, 원본 태그와 substring 충돌 없음).
#
# 완료 후 원본 M4(같은 seed84/fold0 checkpoint 필요 — 없으면 m4_pma_3seed_kfold_array_hpc.sh의
# seed84 fold0 하나만 --array=2로 재제출)와 internal/external c-index 점추정치만 비교.
#
# 제출: sbatch sbatch/final_m1m4_3seed_kfold_hpc/m4_pma_no_coattn_kfold_hpc.sh
# (seed/fold 바꾸려면 아래 SEED=/FOLD= 두 줄만 수정)

cd /pub/wonseukl/Path-ViT/

mkdir -p paper/.hpc

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

SEED=84
FOLD=0
N_FOLDS=5

log="paper/.hpc/train_tcga_seed${SEED}_PMA_uni2native_INT1500_SS_AUX_STG_R_NOCOATTN_DISP_COX_ADD_kfold5_fold${FOLD}.log"

echo "=== M4/PMA-NOCOATTN(co-attention 없음, 4관점 단순평균) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u ./train.py --PMA --dataset tcga --external --seed "${SEED}" \
    --rna-genes literature_1500_intersection \
    --backbone uni2native \
    --clinical-staging --clinical-margin \
    --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 \
    --combine-mode cox_add \
    --no-coattn \
    --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0831_m4pma_no_coattn_1run 2>&1 | tee "${log}"
echo "=== M4/PMA-NOCOATTN seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
