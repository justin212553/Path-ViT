"""
MahmoodLab 공식 UNI2-h feature(256px@20x, 공식 스펙)를 TCGA-PAAD/CPTAC-PDAC에 대해
다운로드한다. gated dataset(MahmoodLab/UNI2-h-features)이라 HuggingFace 계정이 사전에
접근 승인을 받아야 하고, .env의 HF_TOKEN이 그 계정의 토큰이어야 한다(UNI2-h 모델 접근에
쓰던 토큰과 같은 계정이면 재사용 가능하나, 이 dataset repo는 별도로 승인받아야 함).

우리 자체 추출 파이프라인(1024px@1.0MPP -> 512 리사이즈, 실효 2.0MPP)이 UNI2-h 공식 학습/검증
스펙(256px@20x, ~0.5MPP)과 4배 어긋난다는 게 확인돼(2026-08-12), "WSI 신호가 거의 없다"는
이번 세션의 반복된 결론이 아키텍처 문제가 아니라 feature 추출 스펙 문제였을 가능성을 이
공식 feature로 직접 검증한다.

사용법: python scripts/download_uni2h_official_features.py
"""
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from utils import load_env

REPO_ID = "MahmoodLab/UNI2-h-features"
FILES = {
    "tcga": "TCGA/TCGA-PAAD.tar.gz",
    "cptac": "CPTAC/cptac_pda.tar.gz",
}
OUT_ROOT = _ROOT / "data" / "uni2h_official_features"


def main():
    load_env()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for tag, filename in FILES.items():
        print(f"=== {tag}: {filename} 다운로드 시작 ===")
        local_path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=filename,
            local_dir=str(OUT_ROOT / "_archives"),
        )
        print(f"  -> 다운로드 완료: {local_path}")
        extract_dir = OUT_ROOT / tag
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"  -> 압축 해제 중: {extract_dir}")
        with tarfile.open(local_path) as tf:
            tf.extractall(extract_dir)
        print(f"  -> 압축 해제 완료")

    for tag in FILES:
        d = OUT_ROOT / tag
        h5_files = list(d.rglob("*.h5"))
        print(f"{tag}: {len(h5_files)}개 .h5 파일 확인 (예: {h5_files[0].name if h5_files else '없음'})")


if __name__ == "__main__":
    main()
