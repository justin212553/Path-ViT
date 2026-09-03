#!/bin/bash
# 2026-09-02: scripts/run_brca_m7_gene_selection_compare_local.sh(seed 42 단일)의 5시드 확장판.
# variance/cox-top1500/cox-FDR0.1 세 RNA 패널을 표준 5시드(42/84/126/168/210)로 각각 돌려,
# seed 42 단일 결과(FDR 패널이 internal은 제일 좋지만 external은 여전히 variance보다 낮음)가
# 시드 하나의 우연인지 확인한다. --fold 없이 --seed만 바꾸므로 매 시드마다 split도 함께
# 바뀐다(brca_common.load_case_table(seed)) — 이 스크립트는 3개 모델의 통계적 유의성 검정이
# 아니라 "어느 패널이 대체로 나은가"를 보는 탐색적 다시드 확인이라 k-fold paired bootstrap
# 수준의 엄밀함은 필요 없다는 판단(사용자 승인, "시드 5개로 늘려서 한번 확인해봐").
#
# 실행: bash scripts/run_brca_m7_gene_selection_compare_5seed_local.sh > .logs/brca_m7_gene_selection_compare_5seed.log 2>&1

set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEEDS=(42 84 126 168 210)
COMMON="--clinical-staging --epochs 100 --patience 20 --group-ts brca_m7_geneselcompare_5seed_0902"

for SEED in "${SEEDS[@]}"; do
  echo "=== variance top1500 seed=${SEED} $(date) ==="
  python -u -m scripts.train_brca_m7 ${COMMON} --seed "${SEED}" --gene-selection variance --n-genes 1500 2>&1 | tail -8

  echo "=== cox top1500 (raw) seed=${SEED} $(date) ==="
  python -u -m scripts.train_brca_m7 ${COMMON} --seed "${SEED}" --gene-selection cox --n-genes 1500 2>&1 | tail -8

  echo "=== cox FDR<0.1 seed=${SEED} $(date) ==="
  python -u -m scripts.train_brca_m7 ${COMMON} --seed "${SEED}" --gene-selection cox --fdr-threshold 0.1 2>&1 | tail -8
done

echo
echo "##################################################"
echo "########## SUMMARY (5-seed mean +- std, internal / external c-index) ###########"
echo "##################################################"
python - <<'PYEOF'
import pandas as pd
from lifelines.utils import concordance_index

SEEDS = [42, 84, 126, 168, 210]
PANELS = {
    "variance_top1500":  "BRCA_M7_TOP1500_STG_EXTTSSBH",
    "cox_top1500_raw":   "BRCA_M7_TOP1500_COXGENE_STG_EXTTSSBH",
    "cox_fdr0.1":        "BRCA_M7_FDR0.1_COXGENE_STG_EXTTSSBH",
}

for name, prefix in PANELS.items():
    internal, external = [], []
    for seed in SEEDS:
        idf = pd.read_csv(f".logs/kfold_preds/brca_{prefix}_seed{seed}.csv")
        edf = pd.read_csv(f".logs/external_preds/brca_{prefix}_seed{seed}.csv")
        internal.append(concordance_index(idf.OS_time, -idf.risk, idf.OS_event))
        external.append(concordance_index(edf.OS_time, -edf.risk, edf.OS_event))
    import numpy as np
    i, e = np.array(internal), np.array(external)
    print(f"{name:20s} internal={i.mean():.4f}+-{i.std():.4f}  external={e.mean():.4f}+-{e.std():.4f}  "
          f"(per-seed internal={['%.3f'%v for v in internal]}, external={['%.3f'%v for v in external]})")
PYEOF

echo
echo "=== ALL DONE $(date) ==="
