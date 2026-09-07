#!/bin/bash
# PAAD ClusterPool(pdac_consistency_1500, 확정 레시피) 결과 pooling + PORPOISE 최종 대비 paired
# bootstrap. sbatch/pma_clusterpool_pdaccons_final_recipe_2seed_kfold_array_hpc.sh 완료 후 실행.
#
# 태그는 train.py의 여러 조건부 접미사 순서를 직접 뽑은 추정치라 먼저 ls로 실제 파일명부터
# 확인한다 — 틀렸으면 아래 MODEL_B를 실제 파일명에 맞게 고쳐서 재실행.
set -e

echo "=== 실제 생성된 CSV 확인 ==="
ls .logs/kfold_preds/tcga_PMA_uni2native_PDACCONS1500*CLUSTERPOOL* 2>/dev/null || echo "(internal 파일 없음 — 아직 안 끝났거나 태그가 다름)"
ls .logs/external_preds/cptac_PMA_uni2native_PDACCONS1500*CLUSTERPOOL* 2>/dev/null || echo "(external 파일 없음 — 아직 안 끝났거나 태그가 다름)"

MODEL_B="PMA_uni2native_PDACCONS1500_CNV_STG_R_MUT_CLUSTERPOOL_COX_ADD_CLR100_NLLSURV4_NLLCOX1"
MODEL_A="PORPOISE_uni2native_PDACCONS1500_CNV_SS_STG_R_MUT_DISP_CLR100_NLLSURV4_NLLCOX1"

echo ""
echo "=== [1/4] ClusterPool internal (5-fold OOF pooled) ==="
python scripts/pool_multiseed_kfold_preds.py --dataset tcga \
    --model "${MODEL_B}" \
    --seeds 84,126 --n-folds 5 --bootstrap 2000

echo ""
echo "=== [2/4] ClusterPool external (CPTAC) ==="
python scripts/pool_multiseed_external_preds.py --dataset cptac \
    --model "${MODEL_B}" \
    --seeds 84,126 --n-folds 5 --bootstrap 2000

echo ""
echo "=== [3/4] Paired bootstrap: PORPOISE(pdac_consistency_1500) vs ClusterPool — internal ==="
python scripts/paired_bootstrap_delta.py --split internal --dataset tcga \
    --model-a "${MODEL_A}" \
    --model-b "${MODEL_B}" \
    --seeds 84,126 --n-folds 5 --bootstrap 2000

echo ""
echo "=== [4/4] Paired bootstrap: PORPOISE(pdac_consistency_1500) vs ClusterPool — external ==="
python scripts/paired_bootstrap_delta.py --split external --dataset cptac \
    --model-a "${MODEL_A}" \
    --model-b "${MODEL_B}" \
    --seeds 84,126 --n-folds 5 --bootstrap 2000
