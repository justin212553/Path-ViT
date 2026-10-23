#!/bin/bash
# 2026-10-22: Vast.ai(또는 아무 Linux GPU 서버)에서 BRCA QC 재실행·MMP·PORPOISE를 돌리기 위한 1회 준비.
# 권장 인스턴스: RTX 3090/4090 1장, RAM 64GB 이상(작업 하나가 centroid 학습 때 ~13GB), 디스크 160GB 이상,
#              이미지 pytorch/pytorch 계열(CUDA 12). 접속 후:
#   cd /workspace && git clone -b qc-rerun https://github.com/justin212553/Path-ViT && cd Path-ViT
#   bash scripts/compute/vast_setup.sh
# 그 전에(또는 직후) 로컬 data/brca_rna_upload.zip(RNA 두 개, 359MB)을 Jupyter 업로드나 scp로 올린다:
#   scp -P <port> data/brca_rna_upload.zip root@<host>:/workspace/Path-ViT/data/
# PORPOISE까지 돌릴 거면 HPC에서 만든 묶음(scripts/compute/hpc_make_bundle.sh)도 올린다:
#   scp -P <port> pathvit_bundle_porpoise.tar root@<host>:/workspace/Path-ViT/
set -e
cd "$(dirname "$0")/../.."
echo "== repo: $(pwd)"
# BRCA 특징은 받는 데 48GB, 변환 중 잠깐 ~95GB. 디스크는 인스턴스 생성 뒤 못 늘리므로 먼저 확인.
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [ ! -d data/patches_tcga_brca/tiles ] && [ "${FREE_GB:-0}" -lt 100 ]; then
  echo "!! 빈 디스크 ${FREE_GB}GB — 100GB 이상 필요. 디스크 160GB 이상으로 인스턴스를 새로 만들 것"; exit 1
fi

pip install -q nystrom-attention lifelines h5py huggingface_hub timm wandb pandas scikit-learn tqdm
pip install -q faiss-cpu   # MMP prototype(k-means)

# 1) BRCA UNI v1 특징 (Hugging Face 공개 데이터셋, ~48GB) -> 슬라이드별 .pt로 변환(~47GB)
if [ ! -d data/patches_tcga_brca/tiles ]; then
  python scripts/_download_brca_hf.py
  cp data/brca_slide_manifest.csv data/brca_slide_manifest.git.csv
  python -m scripts.prepare_brca_data
  # git의 manifest와 같아야 한다(다르면 같은 코호트가 아님 — 멈추고 확인)
  python - <<'PY'
import pandas as pd
a = pd.read_csv("data/brca_slide_manifest.git.csv"); b = pd.read_csv("data/brca_slide_manifest.csv")
assert set(a.slide_id) == set(b.slide_id), f"manifest 불일치: git {len(a)} vs 새로 만든 것 {len(b)}"
print("manifest 일치:", len(b), "slides")
PY
  rm -rf data/raw_brca_hf   # 변환 뒤 원본은 지워 디스크 확보
fi

# 2) git에 없는 BRCA RNA 두 개(z-score 본 + fold-safe 재정규화용 원값). 로컬 data/brca_rna_upload.zip을
#    Jupyter로 /workspace/Path-ViT/data/ 에 올려 두면 여기서 푼다.
if [ -f data/brca_rna_upload.zip ] && { [ ! -f data/rna_brca.csv ] || [ ! -f data/rna_brca_raw_log2.csv ]; }; then
  python -m zipfile -e data/brca_rna_upload.zip .
fi
for f in data/rna_brca.csv data/rna_brca_raw_log2.csv; do
  [ -f "$f" ] || { echo "!! $f 없음 — 로컬 data/brca_rna_upload.zip을 data/ 에 올릴 것"; exit 1; }
done

# 3) MMP 공식 코드 + 우리 split(instcvqc)
if [ ! -d mmp/src ]; then
  git clone https://github.com/mahmoodlab/MMP mmp
fi
python -m scripts.prepare_mmp_brca_data --mmp-root mmp/src --mmp-dataroot data/mmp_brca_dataroot \
  --protocol instcvqc --seeds 84,126,42,168,210

# 4) PORPOISE 묶음(선택)
if [ -f pathvit_bundle_porpoise.tar ]; then
  tar -xf pathvit_bundle_porpoise.tar
  echo "PORPOISE 묶음 풀림"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv
free -g | head -2
echo "== 준비 끝. 다음: bash scripts/compute/vast_run_all.sh"
