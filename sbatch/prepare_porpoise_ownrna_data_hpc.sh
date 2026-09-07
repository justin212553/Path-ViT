#!/bin/bash
#SBATCH --job-name=PVT-porpoise-ownrna-prep
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/prepare_porpoise_ownrna_data_%j.log

# 2026-09-06: CPTAC을 PORPOISE의 external cohort로 붙이기 위한 1회성 데이터 준비.
#
# PORPOISE 공식 원본 CSV(porpoise/datasets_csv/tcga_paad_all_clean.csv.zip, RNA 전처리 비공개)엔
# CPTAC이 아예 없어서(data/extract_rna_porpoise_official.py 참조), TCGA 쪽도 우리 자체 RNA
# 파이프라인(data/extract_rna_clinical.py — log2(FPKM-UQ+1), protein-coding 19,962유전자,
# TCGA/CPTAC 헤더 완전 일치 확인됨)으로 다시 만들어서 두 코호트가 항상 같은 유전자
# ID/전처리를 쓰도록 한다(사용자 결정 — 유전자 ID/스케일 불일치로 인한 배치 이펙트가
# "일반화 실패"처럼 보이는 걸 방지).
#
# 주의: porpoise/datasets_csv/tcga_paad_all_clean.csv.zip을 실제로 덮어쓴다(원본은
# tcga_paad_all_clean.official_backup.csv.zip으로 자동 백업) — 기존 seed1/84
# PORPOISE-원본-RNA 재현 결과(pooled C≈0.596~0.60)는 이미 별도 results_dir에 저장돼 있어
# 안전하지만, 앞으로 이 CSV를 쓰는 어떤 학습도 own-RNA 버전을 쓰게 됨.
#
# 순서: 1) TCGA own-RNA CSV+splits 생성 2) filter_available_slides.py로 실제 pt_files와
# 대조해 필터링 3) CPTAC external CSV 생성.
#
# 완료 후 확인:
#   unzip -p porpoise/datasets_csv/tcga_paad_all_clean.csv.zip | head -1 | tr ',' '\n' | wc -l
#   unzip -p porpoise/datasets_csv/cptac_paad_external_clean.csv.zip | head -1 | tr ',' '\n' | wc -l
#   (두 값이 거의 같아야 함 — 유전자 컬럼 수 + 메타데이터 컬럼 수)
#
# 제출: sbatch sbatch/prepare_porpoise_ownrna_data_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

echo "=== 1) TCGA-PAAD own-RNA genomic CSV + 5-fold splits 생성: $(date) ==="
python -u -m scripts.prepare_porpoise_paad_data_ownrna

echo "=== 2) true-ResNet50 pt_files 존재 여부로 필터링: $(date) ==="
cd porpoise
python -u filter_available_slides.py --pt-files-dir /pub/wonseukl/Path-ViT/data/porpoise_style_features/tcga/pt_files
cd ..

echo "=== 3) CPTAC external genomic CSV 생성: $(date) ==="
python -u -m scripts.prepare_porpoise_cptac_external_data

echo "=== 완료: $(date) ==="
