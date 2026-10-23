"""
2026-10-22: Kaggle에서 PAAD QC 재실행 한 파트를 돌리는 커널 본체(스크립트형 커널).
scripts/compute/make_kaggle_kernels.py가 PART·MODELS를 채워 파트별 폴더로 복사하고 `kaggle kernels push`한다.

하는 일
 1. GitHub qc-rerun 브랜치를 /tmp/Path-ViT로 받는다(작업 폴더 /kaggle/working은 출력 저장용으로만 쓴다).
 2. Kaggle 데이터셋(path-vit-uni2native-paad, 선택: -extra)의 특징 파일을 data/patches_{tcga,cptac}로
    슬라이드 단위 심볼릭 링크로 합친다(읽기 전용 입력 두 개를 한 트리로 보이게).
 3. scripts.compute.qc_jobs로 이 파트의 작업을 돌린다. 예측 CSV·done 표시·로그는 끝날 때마다
    /kaggle/working/out/.logs/로 복사된다. 시간 예산(기본 11시간)을 넘기면 새 작업을 시작하지 않는다.
"""
import glob
import os
import subprocess
import sys
from pathlib import Path

PART = "__PART__"
MODELS = "__MODELS__".split()
WORKERS = int("__WORKERS__")
BUDGET_H = float("__BUDGET__")
BRANCH = "qc-rerun"
REPO = "https://github.com/justin212553/Path-ViT"
WORK = Path("/tmp/Path-ViT")
OUT = Path("/kaggle/working/out")


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def find_root(name):
    """/kaggle/input 아래에서 slide_index_task0.csv와 tiles/를 같이 가진 patches_<name> 폴더들."""
    hits = []
    for idx in glob.glob("/kaggle/input/**/slide_index_task0.csv", recursive=True):
        d = Path(idx).parent
        if name in str(d) and (d / "tiles").is_dir():
            hits.append(d)
    return hits


def link_features():
    for ds in ("tcga", "cptac"):
        roots = find_root(f"patches_{ds}")
        extra = [Path(p).parent for p in glob.glob(f"/kaggle/input/**/extra_{ds}/tiles", recursive=True)]
        if not roots:
            sys.exit(f"patches_{ds} 입력을 찾지 못함 — 데이터셋이 커널에 붙었는지 확인")
        dst = WORK / "data" / f"patches_{ds}"
        (dst / "tiles").mkdir(parents=True, exist_ok=True)
        for f in roots[0].glob("slide_index_task*.csv"):
            (dst / f.name).symlink_to(f)
        n = 0
        for src in [roots[0]] + extra:
            for sd in (src / "tiles").iterdir():
                link = dst / "tiles" / sd.name
                if not link.exists():
                    link.symlink_to(sd, target_is_directory=True)
                    n += 1
        print(f"patches_{ds}: {n} slides linked from {[str(r) for r in [roots[0]] + extra]}", flush=True)


def main():
    sh(f"git clone --depth 1 -b {BRANCH} {REPO} {WORK}")
    os.chdir(WORK)
    sh("pip install -q nystrom-attention lifelines")
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    link_features()
    OUT.mkdir(parents=True, exist_ok=True)
    sh(f"{sys.executable} -m scripts.compute.qc_jobs --cohort paad --models {' '.join(MODELS)} --part {PART} "
       f"--workers {WORKERS} --budget-hours {BUDGET_H} --mirror {OUT}")


if __name__ == "__main__":
    main()
