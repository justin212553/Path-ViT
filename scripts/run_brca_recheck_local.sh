#!/bin/bash
# 2026-08-31 밤 계획 — BRCA는 이 프로젝트의 핵심 논점이 아니다(핵심은 여전히 PAAD, 사용자
# 결정) — seed42 파일럿만 채우고 다른 시드는 더 이상 쫓지 않는다. BRCA에서 아직 없는 건
# 이 2개뿐(나머지는 이미 있음: M7 seed42/84, PMA(concat) seed42/84, PORPOISE seed84 — paper/
# brca 로그 및 models/checkpoint 참조).
#   1) PMA(concat) + rna-aux-weight=0, seed42 — RNA aux가 정말 아무것도 안 하는지, 코호트가
#      작아서(PAAD) 그렇게 보였던 건지 BRCA 스케일에서 확인.
#   2) PORPOISE(ABMIL+Bilinear, multicomponent 없음) seed42 — 이미 있는 PMA(concat)/M7의
#      seed42 결과와 매칭되는 마지막 조각.
#
# 사용법: PathViT-ray conda env에서
#   bash scripts/run_brca_recheck_local.sh
set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

echo "=== [1/2] PMA(concat) + rna-aux-weight=0, seed=42 ==="
python -u -m scripts.train_brca_m4 --seed 42 --rna-aux-weight 0 --external-tss none \
    --group-ts 0831_recheck 2>&1 | tee .logs/train_brca_m4_noaux_seed42.log

echo "=== [2/2] PORPOISE(ABMIL+Bilinear) seed=42 ==="
python -u -m scripts.train_brca_porpoise --seed 42 --external-tss none \
    --group-ts 0831_recheck 2>&1 | tee .logs/train_brca_porpoise_seed42.log

echo "=== 전부 완료: $(date) ==="
