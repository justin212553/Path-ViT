#!/bin/bash
#SBATCH --job-name=PVT-BRCA-M4-arr
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/brca_m4_multiseed_kfold_array_%a.log

# 2026-09-01: BRCA M4(ViT_PMA, PMA_EX_SS_AUX 레시피)를 PAAD와 동일한 paper-spec 프로토콜
# (2seed x 5fold)로 검증한다 — 기존 BRCA "3시드" 비교(findings_backlog.md)는 전부 split_seed
# (데이터 train/val/test 분할만 바꿈)였지 모델 가중치 초기화 seed가 아니었다는 걸 사용자가
# 지적, 재확인 결과 사실로 확인됨(scripts/brca_common.py::load_case_table가 cfg.data.seed=
# cfg.train.seed=args.seed로 fold 배정과 model init을 항상 같은 값으로 묶어 썼음 — PAAD의
# 표준 2seed(84/126) 프로토콜도 사실 같은 방식이라 이 자체는 문제 아니지만, BRCA는 애초에
# k-fold 자체가 없어 "다시드"라 부를 반복측정이 split 하나짜리 6:2:2뿐이었다).
#
# scripts/brca_common.py::load_case_table_kfold(신규, data/dataset.py::_kfold_case_split과
# 동일 알고리즘 재현)로 진짜 k-fold를 추가하고 scripts/train_brca_m4.py --fold/--n-folds로
# 노출했다 — fold=0..4를 pooled out-of-fold로 이어붙이면 internal 표본이 코호트 전체(916명,
# institution BH 142명 제외)로 늘어나고, seed=84/126 두 번 반복하면 PAAD와 동일한 반복측정
# 구조가 된다.
#
# [institution(BH) external holdout 미사용] 2026-08-31 결정(train_brca_m4_internal_hpc.sh
# 참조) — BH 인구가 event rate 등에서 나머지와 너무 달라 신뢰하기 어렵다고 판단해 그 뒤로
# --external-tss none으로 꺼왔다. 이번 k-fold 검증도 동일하게 internal만 본다(916+142=1058
# 전체를 internal 6:2:2/kfold 풀로 씀).
#
# [주의] 로컬 스모크 테스트(1 epoch, seed84/fold0)로 wiring만 확인함 — 전체 30epoch 실행
# 시간은 원본 단일-런(전체 1058명, --time=24:00:00) 기준과 fold당 train 크기가 비슷해
# 시간대는 유사할 것으로 예상하나 실측은 안 됨. 10개 array task가 free-gpu 파티션 여유에 따라
# 동시에 안 돌 수 있다 — 노드 여유가 부족하면 실제 완료까지 24h*(대기 배수)가 걸릴 수 있음.
#
# SLURM_ARRAY_TASK_ID(0~9) -> seed_idx = id/5, fold = id%5.
#
# 완료 후(10개 fold 로그 확인, .logs/kfold_preds/·.logs/external_preds/에 CSV 10개씩 있는지 확인):
#   python scripts/pool_multiseed_kfold_preds.py --dataset brca \
#       --model BRCA_PMA_TOP1500_SS_AUX --seeds 84,126 --n-folds 5 --bootstrap 2000
#   (M7과의 paired 비교는 scripts/paired_bootstrap_delta.py 참조 — .logs/kfold_preds/의
#   두 모델 CSV를 그대로 입력으로 쓸 수 있음)
#
# 제출: sbatch sbatch/brca_m4_multiseed_kfold_array_hpc.sh

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

log=".logs/train_brca_m4_seed${SEED}_fold${FOLD}of${N_FOLDS}.log"

echo "=== BRCA M4 seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m scripts.train_brca_m4 --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" \
    --external-tss none --group-ts 0901_brca_m4_multiseed_kfold_hpc 2>&1 | tee "${log}"
echo "=== BRCA M4 seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
