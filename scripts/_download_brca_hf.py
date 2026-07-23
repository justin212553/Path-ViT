"""TCGA-BRCA WSI feature(UNI backbone, 1024dim) + patch 좌표를 HF Dataset(Dearcat/CPathPatchFeature)에서
다운로드한다 - "WSI 브랜치가 표본만 늘리면 공간 신호를 학습하는가"를 검증하기 위한 대규모(1131슬라이드)
보조 코호트 준비 작업(findings_backlog.md 관련 논의 참조).
"""
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Dearcat/CPathPatchFeature",
    repo_type="dataset",
    allow_patterns=["brca/uni/pt_files/*", "brca/patches/*"],
    local_dir="data/raw_brca_hf",
)
print("DOWNLOAD_DONE:", path)
