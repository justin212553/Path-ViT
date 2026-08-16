#!/bin/bash
# M1_uni2_DISP_NOVIT_COORD_CAT_SHUF — coord-embed-concat 대조군. forward마다 patch 순서를
# 무작위로 섞은 coords를 coord_embed에 넣어(coord_fusion 구조/파라미터는 동일, "이 패치가
# 실제로 어디 있는가"라는 대응만 파괴) 개선이 진짜 위치 정보 때문인지 capacity 효과인지 구분.
# fold0 파일럿: test_c_index=0.6446 (진짜 좌표 버전 0.6413과 거의 동일, 오히려 미세하게 높음).
set -e
cd "$(dirname "$0")/.."

SEED=42
N_FOLDS=5

for ((FOLD=1; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_M1_uni2_DISP_NOVIT_COORD_CAT_SHUF_kfold5_fold${FOLD}.log"
    echo "=== M1+coord-embed-concat-shuffle seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
    python -u ./train.py --M1 --skip-patch-vit --attn-dispersion --coord-embed --coord-embed-concat --coord-embed-shuffle \
        --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0815m1_coordcat_shuffle_seed42_kfold5_local 2>&1 | tee "${log}"
    echo "=== M1+coord-embed-concat-shuffle seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
done
