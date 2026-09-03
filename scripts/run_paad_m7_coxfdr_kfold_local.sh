#!/bin/bash
# 2026-09-03: PAAD M7 RNA 패널 leak 재검증 — literature_1500_intersection(leaky, findings_backlog.md
# 2026-09-02 항목)과 pathway8(leak-free, 문헌 큐레이션)을 비교했던 2seed(84/126)x5fold 레시피
# (--clinical-staging --clinical-margin --combine-mode cox_add, model_prefix *_STG_R_COX_ADD)를
# 그대로 재사용하되, --rna-genes만 literature_fdr0.1_tcga_only(생존 라벨 기반 Cox+BH-FDR,
# single-cohort라 CPTAC label 미참조 — external leak 없음)로 바꿔서 돌린다.
#
# [주의] 이 패널도 여전히 _train_case_ids_single()의 고정 단일 6:2:2 split으로 유전자를 뽑는다
# (fold-aware 재선정 아님) — literature_1500_intersection과 동일한 종류의 internal
# fold-경계 leak 구조는 남아있다. 다만 FDR로 걸러낸 소수 유전자만 쓰면 그 구조적 overlap이
# 있어도 각 유전자가 특정 환자 집단에 과적합될 여지 자체가 줄어(레퍼런스: BRCA에서 동일 방식
# 검증 시 노이즈 유전자 비율이 top1500 대비 크게 줄었음, findings_backlog.md) internal 인플레이션이
# 줄어들 수 있다는 게 이번 실험의 가설 — external(둘 다 CPTAC 라벨 미참조라 완전 leak-free)과
# 비교해 internal-external gap이 얼마나 좁혀지는지가 핵심 지표.
#
# 실행: bash scripts/run_paad_m7_coxfdr_kfold_local.sh > .logs/paad_m7_coxfdr_kfold.log 2>&1

set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

SEEDS=(84 126)
N_FOLDS=5
COMMON="--M7 --dataset tcga --external --rna-genes literature_fdr0.1_tcga_only --clinical-staging --clinical-margin --combine-mode cox_add --group-ts 0903_paad_m7_coxfdr_kfold_local"

for SEED in "${SEEDS[@]}"; do
  for FOLD in 0 1 2 3 4; do
    echo "=== PAAD M7 coxfdr0.1 seed=${SEED} fold=${FOLD}/${N_FOLDS} $(date) ==="
    python -u train_light.py ${COMMON} --seed "${SEED}" --fold "${FOLD}" --n-folds "${N_FOLDS}" 2>&1 | tail -8
  done
done

echo
echo "##################################################"
echo "########## SUMMARY: internal / external pooled c-index ###########"
echo "##################################################"
echo "--- internal (tcga k-fold pooled) ---"
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M7_EXTfdr0.1_STG_R_COX_ADD --seeds 84,126 --n-folds 5
echo
echo "--- external (cptac, 전체 코호트) ---"
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M7_EXTfdr0.1_STG_R_COX_ADD --seeds 84,126 --n-folds 5

echo
echo "=== ALL DONE $(date) ==="
