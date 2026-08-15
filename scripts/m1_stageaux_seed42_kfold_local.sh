#!/bin/bash
# M1_uni2_STG_DISP_AUX2_NOVIT — M1(WSI 단독, self-ABMIL, skip-patch-vit, DISP)에
# --stage-aux-weight(T-stage/grade 보조과제, models/stage_predictor.py) 추가.
# 2026-08-14: M1 seed42 fold0의 val_c_index가 epoch1에서 멈추고(best_val=0.382) 그 이후
# 내내 0.4 밑으로 정체하던 문제(train_c는 정상적으로 오르는데 val만 처음부터 반대 방향)에
# 대한 대응 — RNA 없이도 WSI 브랜치에 더 촘촘한 학습 신호를 주는 유일한 보조과제.
# fold0 파일럿: test_c_index(best ckpt) 0.4137->0.5298로 개선.
set -e
cd "$(dirname "$0")/.."

SEED=42
N_FOLDS=5

for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    log=".logs/train_tcga_seed${SEED}_M1_uni2_STG_DISP_AUX2_NOVIT_kfold5_fold${FOLD}.log"
    echo "=== M1+stage-aux(uni2,skip-patch-vit,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Start: $(date) ==="
    python -u ./train.py --M1 --skip-patch-vit --attn-dispersion --stage-aux-weight 1.0 \
        --dataset tcga --external --seed "${SEED}" \
        --backbone uni2 \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --group-ts 0814m1_stageaux_seed42_kfold5_local 2>&1 | tee "${log}"
    echo "=== M1+stage-aux(uni2,skip-patch-vit,DISP) seed=${SEED} fold=${FOLD}/${N_FOLDS} Complete: $(date) ==="
done
