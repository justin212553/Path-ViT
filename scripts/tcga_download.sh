#!/bin/bash
#SBATCH --job-name=tcga_paad_download   # Slurm 잡 이름
#SBATCH --partition=free                # ★핵심: 비용 차감 없는 무료 CPU 파티션 지정
#SBATCH --nodes=1                       # 단일 노드 점유
#SBATCH --ntasks-per-node=1             # 단일 태스크
#SBATCH --cpus-per-task=8               # gdc-client 멀티스레드 병렬 처리를 위해 코어 8개 할당
#SBATCH --mem=16G                       # 네트워크 버퍼 소화용 메모리 16GB
#SBATCH --time=2-00:00:00               # Walltime 2일 지정 (free 상한선인 3일 이내 안착)
#SBATCH --output=./.logs/gdc_download.log     # 다운로드 로그 실시간 기록 파일

# ── 1. 경로 정의 ─────────────────────────────────────────────────────────────
MANIFEST_PATH="./scripts/gdc_manifest.txt"  # 아까 Portal에서 받아둔 매니페스트 경로
DOWNLOAD_DIR="./data/tcga_paad_wsi"                  # SVS 저장할 폴더
GDC_CLIENT="./.bin/gdc-client"                       # 위에서 세팅한 바이너리 경로

mkdir -p $DOWNLOAD_DIR

echo "====================================================="
echo "TCGA-PAAD WSI 다운로드 프로세스 가동 (free partition)"
echo "시작 시간: $(date)"
echo "====================================================="

# ── 2. GDC Client 매니페스트 기반 다운로드 실행 ──────────────────────────────
# -n 8: 코어 8개를 모두 쥐고 8개 파일을 동시 병렬 다운로드하여 속도 극대화
$GDC_CLIENT download -m $MANIFEST_PATH -d $DOWNLOAD_DIR -n 8 --no-related-files --no-annotations

echo "====================================================="
echo "다운로드 프로세스 종료 시간: $(date)"
echo "====================================================="