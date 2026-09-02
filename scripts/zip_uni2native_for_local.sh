#!/bin/bash
# 2026-09-02: HPC 전용 — uni2native WSI feature/좌표(data/patches_{tcga,cptac}/tiles/<slide>/
# {features_uni2native.pt,coords_uni2native.pt})가 로컬엔 0개(직접 재확인)라, HPC에서 이 두
# 파일만 골라 zip으로 묶어 로컬로 옮긴다 — 전체 tile 트리(jpg 등 포함)를 통째로 옮기면 훨씬
# 크므로, feature/좌표 파일만 골라 담는다.
#
# 목적: train_hdp_pretrain_cluster.py(feature_backbone="uni2native")를 HPC GPU 자리가 안 날 때
# 로컬에서 대신 돌리기 위함(2026-09-02, 새로 학습한 해상도 보정 head 검증용).
#
# 실행(HPC, Path-ViT 루트에서): bash scripts/zip_uni2native_for_local.sh
# 결과: uni2native_for_local.zip (같은 디렉토리) — 로컬로 다운받은 뒤 프로젝트 루트에서
#   unzip uni2native_for_local.zip
# 하면 data/patches_tcga/tiles/.../features_uni2native.pt 형태로 그대로 풀린다(zip이 상대경로를
# 그대로 담으므로 별도 이동/복사 불필요).

set -e
cd "$(dirname "$0")/.."

OUT="uni2native_for_local.zip"
rm -f "$OUT"

echo "대상 파일 목록 수집 중..."
find data/patches_tcga/tiles data/patches_cptac/tiles \
    \( -name "features_uni2native.pt" -o -name "coords_uni2native.pt" \) \
    > /tmp/uni2native_filelist.txt
N=$(wc -l < /tmp/uni2native_filelist.txt)
echo "  대상 파일 ${N}개"

if [ "$N" -eq 0 ]; then
    echo "[경고] uni2native 파일을 하나도 못 찾았습니다 — 경로/파일명을 확인하세요."
    exit 1
fi

zip -q "$OUT" -@ < /tmp/uni2native_filelist.txt
rm -f /tmp/uni2native_filelist.txt

echo "완료: ${OUT} ($(du -h "$OUT" | cut -f1))"
