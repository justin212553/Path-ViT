#!/bin/bash
# 2026-10-22: HPC 로그인 노드에서 한 번 실행(GPU 불필요). HPC에만 있는 파일을 두 개의 tar로 묶는다.
#   ssh hpc3.rcic.uci.edu
#   cd /pub/wonseukl/Path-ViT && git fetch origin && git checkout origin/qc-rerun -- scripts/compute
#   bash scripts/compute/hpc_make_bundle.sh
# 만들어지는 것(~/ 아래):
#   pathvit_missing9.tar          로컬에 없는 TCGA DX 슬라이드 9장의 uni2native 특징
#                                 -> 로컬로 받아서: python -m scripts.compute.stage_kaggle_extra pathvit_missing9.tar
#   pathvit_bundle_porpoise.tar   PORPOISE 코드·split·CSV + ResNet50 특징(QC split 포함)
#                                 -> Vast 서버의 /workspace/Path-ViT/ 로 scp (vast_setup.sh가 풂)
# 받기(로컬 PowerShell/터미널):  scp hpc3.rcic.uci.edu:~/pathvit_missing9.tar .
set -e
cd /pub/wonseukl/Path-ViT
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Path-ViT

# 1) 빠진 9장
MISSING=(
  TCGA-IB-7886-01Z-00-DX1.451DCD29-42FF-4A7C-885A-114F33ABCC06
  TCGA-IB-A6UG-01Z-00-DX1.2EF1F819-ED99-40A1-A87B-79C1952E8055
  TCGA-IB-A5SQ-01Z-00-DX1.790002F9-0B3B-44F6-A888-82E87FC78B73
  TCGA-IB-AAUN-01Z-00-DX1.70F5AFF8-FECD-4BEE-AD00-E03F05D29B58
  TCGA-IB-AAUR-01Z-00-DX1.97C45E09-EC88-4543-A854-63C87BAFA782
  TCGA-L1-A7W4-01Z-00-DX5.C7876997-F7D0-48EC-8869-AB9457FFF6EB
  TCGA-RB-AA9M-01Z-00-DX1.3A709BA7-0E96-49D2-93A5-5193E6156E64
  TCGA-US-A779-01Z-00-DX1.67E123A6-0A8D-48F9-9A4B-1213E411B9B1
  TCGA-S4-A8RP-01Z-00-DX1.A4A3DC84-98EC-49C5-860F-D3C4E16A0E29
)
FILES=()
for s in "${MISSING[@]}"; do
  for f in features_uni2native.pt coords_uni2native.pt; do
    p="data/patches_tcga/tiles/$s/$f"
    if [ -f "$p" ]; then FILES+=("$p"); else echo "!! HPC에도 없음: $p"; fi
  done
done
# 2026-10-22 확인: 9장은 HPC에도 특징이 없다 -> 원래 학습에서도 빠져 있었음(데이터셋이 특징 없는 슬라이드를 건너뜀).
# 그래서 Kaggle 데이터셋(268장)이 HPC와 같은 구성이다. 있으면 묶고, 없으면 건너뛴다.
if [ ${#FILES[@]} -gt 0 ]; then
  tar -cf ~/pathvit_missing9.tar "${FILES[@]}"
  echo "pathvit_missing9.tar: ${#FILES[@]} files, $(du -h ~/pathvit_missing9.tar | cut -f1)"
else
  echo "9장 모두 HPC에도 없음 — 건너뜀(보조 데이터셋 불필요)"
fi

# 2) PORPOISE QC split 만들기(이미 있으면 덮어써도 같은 내용) + 묶기
python -m scripts.prepare_porpoise_paad_n110 --qc
python -m scripts.prepare_porpoise_paad_n110 --official-geneset --qc
tar -cf ~/pathvit_bundle_porpoise.tar \
  --exclude='porpoise/results*' --exclude='porpoise/data_root_true_resnet50' --exclude='porpoise/.git' \
  porpoise data/porpoise_style_features
echo "pathvit_bundle_porpoise.tar: $(du -h ~/pathvit_bundle_porpoise.tar | cut -f1)"
