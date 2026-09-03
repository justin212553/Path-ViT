#!/bin/bash
# 2026-09-02: BRCA M7 RNA 유전자 패널 비교 — 고분산(variance, 기존) vs Cox top-1500(잡음 많음,
# 실측 확인 — RBMY1E/TSPY1 등 Y염색체 유전자·후각수용체가 상위권을 채움) vs Cox FDR<0.1(121개,
# 노이즈 필터링). 셋 다 --clinical-staging(T/N/M) 켜고 단일 seed(42)로 빠르게 비교만 한다
# (통계적 다시드 검증이 아니라 "어느 패널이 그나마 나은지" 탐색 목적).
#
# 실행: bash scripts/run_brca_m7_gene_selection_compare_local.sh > .logs/brca_m7_gene_selection_compare.log 2>&1

set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEED=42
COMMON="--seed ${SEED} --clinical-staging --epochs 100 --patience 20 --group-ts brca_m7_geneselcompare_0902"

echo "=== [1/3] variance top1500 (기존 baseline) $(date) ==="
python -u -m scripts.train_brca_m7 ${COMMON} --gene-selection variance --n-genes 1500 2>&1 | tail -15

echo
echo "=== [2/3] cox top1500 (raw, 잡음 포함) $(date) ==="
python -u -m scripts.train_brca_m7 ${COMMON} --gene-selection cox --n-genes 1500 2>&1 | tail -15

echo
echo "=== [3/3] cox FDR<0.1 (121개, 잡음 필터링) $(date) ==="
python -u -m scripts.train_brca_m7 ${COMMON} --gene-selection cox --n-genes 121 2>&1 | tail -15

echo
echo "##################################################"
echo "########## SUMMARY (internal test c-index / HR / log-rank p, seed=${SEED}) ###########"
echo "##################################################"
for TAG in "brca_BRCA_M7_TOP1500_STG_EXTTSSBH_seed${SEED}" \
           "brca_BRCA_M7_TOP1500_COXGENE_STG_EXTTSSBH_seed${SEED}" \
           "brca_BRCA_M7_TOP121_COXGENE_STG_EXTTSSBH_seed${SEED}"; do
  echo "--- ${TAG} ---"
  python -c "
import pandas as pd
from lifelines.utils import concordance_index
df = pd.read_csv('.logs/kfold_preds/${TAG}.csv')
print(f'  n={len(df)} events={int(df.OS_event.sum())} c_index={concordance_index(df.OS_time, -df.risk, df.OS_event):.4f}')
"
done

echo
echo "=== ALL DONE $(date) ==="
