#!/bin/bash
#SBATCH --job-name=PVT-porpoise-feat
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/pub/wonseukl/Path-ViT/.logs/extract_porpoise_style_features_%j.log

# 2026-09-05: porpoise/inputs/tcga_paad_20x_features/pt_files/의 기존 .pt가 PORPOISE 원본이
# 아니라 우리 UNI2 feature(1536차원)를 재포장한 것이었음을 확인(진짜 원본은 ImageNet ResNet50
# truncated, 1024차원) — "PORPOISE가 우리와 다른 전처리로 신호를 뽑아낸 게 아닐까"를 직접
# 검증하기 위해 진짜 원본 스펙으로 재추출한다.
#
# 2026-09-05(2차 수정): 1차 버전은 패치 위치를 UNI2-h 공식 추출(data/uni2h_official_features/
# tcga/*.h5)의 coords에서 재사용했는데, 그 컬렉션이 203개 슬라이드 전부 DX(진단용/영구절편)뿐
# 이라 PORPOISE 공식 CSV가 실제로 쓰는 377개 슬라이드(DX+TS/BS=냉동절편·생검)의 절반 이상을
# 못 커버해 HPC 실행이 없는 파일 때문에 크래시했다. 확인 결과 이건 우리 본 실험(uni2native
# 백본, data/patches_tcga/tiles/, 466개 슬라이드 DX+TS+BS 다 포함)과는 무관한 문제 —
# uni2h_official_features는 이 프로젝트에서 거의 안 쓰이는 별도 다운로드였을 뿐, uni2native는
# 원래부터 전체 슬라이드 유형을 다 썼다. 이번엔 UNI2-h에 기대지 않고 PORPOISE 공식 CSV
# (porpoise/datasets_csv/tcga_paad_all_clean.csv.zip)가 실제로 필요로 하는 377개 슬라이드를
# 직접 기준 삼아 독립적으로 tissue segmentation(HSV saturation + Otsu, cv2/skimage 없이
# numpy+scipy로 재구현)을 해서 전부 커버한다. WSI 원본(.svs) 377개 전부 로컬 확인 완료
# (data/tcga_paad_wsi/).
#
# 완료 후: data/porpoise_style_features/tcga/pt_files/*.pt (N x 1024, ImageNet ResNet50
# layer3 truncated) 377개 생성 확인.
#
# 제출: sbatch sbatch/extract_porpoise_style_features_hpc.sh

cd /pub/wonseukl/Path-ViT/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

echo "=== PORPOISE-style(ImageNet ResNet50 truncated, 1024-dim, 독립 tissue segmentation) feature 추출 Start: $(date) (job ${SLURM_JOB_ID}, node $(hostname)) ==="
python -u -m data.extract_porpoise_style_features
echo "=== PORPOISE-style feature 추출 Complete: $(date) ==="
