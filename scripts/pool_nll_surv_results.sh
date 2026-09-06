#!/bin/bash
# 2026-09-06: PORPOISE/PMA/PMA+ClusterPool nll_surv loss-swap 실험(2seed x 5fold) 전부의
# internal(tcga)+external(cptac) pooled 지표를 한 번에 뽑는다. GPU 불필요, HPC 로그인 노드에서
# 바로 실행 가능 — .logs/kfold_preds/의 CSV만 읽는 순수 집계 스크립트
# (scripts/pool_multiseed_kfold_preds.py).
#
# --dataset tcga: internal(같은 TCGA-PAAD 코호트, out-of-fold pooled)
# --dataset cptac: external(학습에 전혀 안 쓴 CPTAC 코호트) — sbatch 스크립트가 전부
#   --dataset tcga --external로 돌아서, 매 fold 학습이 끝날 때마다 그 fold의 체크포인트로
#   CPTAC 전체를 평가한 예측도 같은 .logs/kfold_preds/ 관례로 저장돼 있다(train.py,
#   external_dataset 변수 기준 pred_path).
#
# 실행 전 확인: sbatch/{porpoise,pma,pma_clusterpool}_nll_surv_loss_10fold_array_hpc.sh 10개
# fold(각 스크립트당 --array=0-9, seed 84+126) 전부 완료돼 있어야 함
# (ls .logs/kfold_preds/tcga_<태그>_seed{84,126}_fold{0..4}of5.csv 10개씩 확인 권장).
#
# 사용법: bash scripts/pool_nll_surv_results.sh

set -e
cd "$(dirname "$0")/.."

SEEDS="84,126"
N_FOLDS=5
BOOTSTRAP=2000

PORPOISE_TAG="PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4"
PMA_TAG="PMA_uni2native_INT1500_CNV_SS_AUX_STG_R_MUT_DISP_COX_ADD_CLR100_NLLSURV4"
PMA_CLUSTERPOOL_TAG="PMA_uni2native_INT1500_CNV_SS_AUX_STG_R_MUT_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4"

run_pool() {
    local label="$1" dataset="$2" tag="$3"
    echo ""
    echo "########## ${label} — ${dataset} ##########"
    python scripts/pool_multiseed_kfold_preds.py --dataset "${dataset}" \
        --model "${tag}" --seeds "${SEEDS}" --n-folds "${N_FOLDS}" --bootstrap "${BOOTSTRAP}"
}

for tag_pair in "PORPOISE:${PORPOISE_TAG}" "PMA:${PMA_TAG}" "PMA+ClusterPool:${PMA_CLUSTERPOOL_TAG}"; do
    label="${tag_pair%%:*}"
    tag="${tag_pair#*:}"
    run_pool "${label}" "tcga" "${tag}"
    run_pool "${label}" "cptac" "${tag}"
done
