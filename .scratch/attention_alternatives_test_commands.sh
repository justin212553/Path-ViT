#!/bin/bash
# 2026-09-05: 나이스트롬 대신/보완할 3가지 패치 간 정보교환 방식 — 한 fold/seed(0/84)씩만 빠르게 비교.
# 오늘 채택한 M4 baseline(pdac_consistency_1500+CNV+mutation+STG+margin+CLR100+uni2native) 위에
# attention 방식만 하나씩 바꿔서 얹는다. 로컬 GPU 사정 봐가며 하나씩 실행할 것 (동시 실행 금지).
#
# 비교 기준(오늘 이미 확인된 같은 레시피의 baseline, 단일 fold0/seed84):
#   internal test c_index = 0.536, external c_index = 0.630 (CLR100, Nystrom 기본값 그대로)

cd /d/wonse/Documents/Job/urban_datalab/PATH-ViT

BASE_ARGS="--M4 --dataset tcga --external --seed 84 --fold 0 --n-folds 5 \
  --backbone uni2native --rna-genes pdac_consistency_1500 --use-cnv --clinical-mutation \
  --clinical-staging --clinical-margin --combine-mode cox_add \
  --clinical-lr-mult 100 --lr-mult-warmup-epochs 10"

echo "=== 1) Nystrom landmark 버그 수정 (--fix-nystrom-landmarks) ==="
python -u ./train.py $BASE_ARGS --fix-nystrom-landmarks 2>&1 | tee .logs/attn_test1_nystromfix_fold0seed84.log

echo "=== 2) kNN 평균 집계, attention 없음 (--knn-mean-agg) ==="
python -u ./train.py $BASE_ARGS --knn-mean-agg 2>&1 | tee .logs/attn_test2_knnmeanagg_fold0seed84.log

echo "=== 3) 클러스터 압축 + 슈퍼토큰 dense attention (--cluster-attn) ==="
python -u ./train.py $BASE_ARGS --cluster-attn --n-clusters 16 2>&1 | tee .logs/attn_test3_clusterattn_fold0seed84.log
