#!/bin/bash
#SBATCH --job-name=PVT-porpoise-feat
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/extract_porpoise_style_features_%j.log

# 2026-09-05: porpoise/inputs/tcga_paad_20x_features/pt_files/의 기존 .pt가 PORPOISE 원본이
# 아니라 우리 UNI2 feature(1536차원)를 재포장한 것이었음을 확인(진짜 원본은 ImageNet ResNet50
# truncated, 1024차원) — "PORPOISE가 우리와 다른 전처리로 신호를 뽑아낸 게 아닐까"를 직접
# 검증하기 위해 진짜 원본 스펙으로 재추출한다.
#
# 2026-09-05(2차 수정, 폐기): 1차 버전은 패치 위치를 UNI2-h 공식 추출(data/uni2h_official_features/
# tcga/*.h5)의 coords에서 재사용했는데 203개 슬라이드(전부 DX)뿐이라 PORPOISE 공식 CSV의 377개
# (DX+TS/BS)를 못 커버했다. 2차 버전은 SVS 원본에서 Otsu tissue segmentation을 직접 재구현했는데,
# 이건 불필요한 중복이었다.
#
# 2026-09-05(3차 수정): uni2native 리타일링 단계(sbatch/preprocess_uni2native_retile_array_hpc.sh)가
# 이미 data/patches_tcga_uni2native/tiles/<slide_id>/*.jpg 에 256px@0.5MPP(PORPOISE 논문 스펙과
# 사실상 동일 해상도) tissue-segmented 패치를 다 만들어 뒀다 — DX+TS+BS 전체(466슬라이드) 커버.
# 그 jpg 픽셀 자체는 인코더와 무관하게 재사용 가능(재사용 불가능한 건 오직
# features_uni2native.pt=UNI2-h 임베딩 자체뿐, 아키텍처가 달라 호환 안 됨). 그래서 SVS/openslide
# 접근도, tissue segmentation 재구현도 다 필요 없어졌다 — 남은 건 그 jpg를 ImageNet ResNet50
# (layer3 truncated)에 통과시키는 GPU forward pass 하나뿐이라 훨씬 가볍고 빠르다(time 12h->6h로
# 하향, 필요시 조정).
#
# 선행 조건: data/patches_tcga_uni2native/tiles/ 가 HPC에 이미 존재해야 함(로컬엔 용량 때문에
# features_uni2native.pt만 동기화돼 있고 jpg 원본은 HPC에만 있음 — 이미 있는 걸로 확인됨).
#
# 완료 후: data/porpoise_style_features/tcga/pt_files/*.pt (N x 1024, ImageNet ResNet50
# layer3 truncated) 377개 생성 확인.
#
# 제출: sbatch sbatch/extract_porpoise_style_features_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

echo "=== PORPOISE-style(ImageNet ResNet50 truncated, 1024-dim, uni2native 타일 재사용) feature 추출 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m data.extract_porpoise_style_features
echo "=== PORPOISE-style feature 추출 Complete: $(date) ==="
