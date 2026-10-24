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
# MMP·PORPOISE 공식 코드가 import하는 것(2026-10-23 Vast에서 빠져 있던 것)
pip install -q transformers opencv-python-headless POT seaborn scikit-survival tensorboardX

# 1) BRCA UNI v1 특징 (Hugging Face 공개 데이터셋, ~48GB) -> 슬라이드별 .pt로 변환(~47GB)
if [ ! -d data/patches_tcga_brca/tiles ]; then
  python scripts/_download_brca_hf.py
  cp data/brca_slide_manifest.csv data/brca_slide_manifest.git.csv
  python -m scripts.prepare_brca_data
fi
if [ -f data/brca_slide_manifest.git.csv ]; then
  # git의 manifest와 같아야 한다(다르면 같은 코호트가 아님 — 멈추고 확인). git 판은 임상 라벨 수정(2026-09-02)
  # 전에 만들어져 TCGA-A7-A0DC(슬라이드 2장)를 더 갖고 있다. 학습 코호트는 임상∩RNA∩manifest라
  # HPC 학습에서도 이 환자는 빠져 있었으므로, 임상 라벨이 있는 환자끼리 비교한다.
  python - <<'PY'
import os
import pandas as pd
c = set(pd.read_csv("data/brca_clinical.csv").case_id)
a = pd.read_csv("data/brca_slide_manifest.git.csv"); b = pd.read_csv("data/brca_slide_manifest.csv")
a = a[a.case_id.isin(c)]
assert set(a.slide_id) == set(b.slide_id), f"manifest 불일치: git(임상 있는 환자) {len(a)} vs 새로 만든 것 {len(b)}"
j = a.merge(b, on="slide_id", suffixes=("_git", "_new"))
bad = j[(j.case_id_git != j.case_id_new) | ((j.OS_time_git - j.OS_time_new).abs() > 1e-6) | (j.OS_event_git != j.OS_event_new)]
assert bad.empty, f"manifest 라벨 불일치 {len(bad)}장: {bad.slide_id.tolist()[:3]}"
miss = [s for s in b.slide_id if not os.path.exists(f"data/patches_tcga_brca/tiles/{s}/features_uni.pt")]
assert not miss, f"특징 파일 없는 슬라이드 {len(miss)}장: {miss[:3]}"
print("manifest 일치:", len(b), "slides,", b.case_id.nunique(), "cases (git 판에서 임상 라벨 없는 환자만 제외)")
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
# 공식 trainer.py 끝의 writer.close()는 정의 안 된 writer를 불러 결과 저장 직전 NameError(2026-10-23). 쓰이지 않으므로 막는다.
sed -i "s/^    writer.close()$/    # writer.close()  # 공식 코드에 writer 정의가 없어 NameError — 쓰이지 않으므로 막음/" mmp/src/training/trainer.py
# 같은 파일 validate_survival: 마지막 배치가 1명이면 attention이 2차원이 돼 np.vstack이 실패(seed 168 fold 2 external).
# 그 배치만 차원을 되살린다 — 모든 배치가 같은 차원인 경우에는 원래와 같다.
sed -i -E "s/^( +)(all_(omic|cross|path)_attn) = np\.vstack\(\2\)$/\1\2 = np.vstack([x[None] if x.ndim == 2 else x for x in \2])/" mmp/src/training/trainer.py
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
