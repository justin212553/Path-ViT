#!/bin/bash
# 2026-09-06: PORPOISE/PMA/PMA+ClusterPool nll_surv loss-swap 실험(2seed x 5fold) 전부의
# internal(tcga)+external(cptac) pooled 지표를 한 번에 뽑는다. GPU 불필요, HPC 로그인 노드에서
# 바로 실행 가능.
#
# internal(tcga): .logs/kfold_preds/의 CSV(scripts/pool_multiseed_kfold_preds.py) — seed 간
#   겹치지 않는 held-out 환자만 골라 평균.
# external(cptac): **.logs/external_preds/의 별도 CSV**, 별도 스크립트
#   (scripts/pool_multiseed_external_preds.py)를 써야 한다 — 처음엔 internal 스크립트에
#   --dataset cptac만 주면 될 줄 알았는데(2026-09-06 시행착오), train.py가 external 예측을
#   아예 다른 디렉토리(.logs/external_preds/)에 저장해서 안 맞았다. external은 코호트 자체가
#   학습에서 완전히 배제되므로 held-out 제약 없이 2seed x 5fold=10개 실행 전부의 예측을 그냥
#   평균한다(내부 로직 차이).
#
# 실행 전 확인: sbatch/{porpoise,pma,pma_clusterpool}_nll_surv_loss_10fold_array_hpc.sh 10개
# fold(각 스크립트당 --array=0-9, seed 84+126) 전부 완료돼 있어야 함.
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
# --surv-loss both(nll_surv+cox 혼합, sbatch/porpoise_both_loss_10fold_array_hpc.sh) — 2026-09-06 추가.
PORPOISE_BOTH_TAG="PORPOISE_uni2native_INT1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1"

run_internal() {
    local label="$1" tag="$2"
    echo ""
    echo "########## ${label} — tcga(internal) ##########"
    python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
        --model "${tag}" --seeds "${SEEDS}" --n-folds "${N_FOLDS}" --bootstrap "${BOOTSTRAP}"
}

run_external() {
    local label="$1" tag="$2"
    echo ""
    echo "########## ${label} — cptac(external) ##########"
    python scripts/pool_multiseed_external_preds.py --dataset cptac \
        --model "${tag}" --seeds "${SEEDS}" --n-folds "${N_FOLDS}" --bootstrap "${BOOTSTRAP}"
}

for tag_pair in "PORPOISE:${PORPOISE_TAG}" "PMA:${PMA_TAG}" "PMA+ClusterPool:${PMA_CLUSTERPOOL_TAG}" "PORPOISE+both-loss:${PORPOISE_BOTH_TAG}"; do
    label="${tag_pair%%:*}"
    tag="${tag_pair#*:}"
    run_internal "${label}" "${tag}"
    run_external "${label}" "${tag}"
done
