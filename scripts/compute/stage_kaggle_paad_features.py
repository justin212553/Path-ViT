"""
2026-10-22: HPC free-gpu가 1주째 안 돌아 PAAD QC 재실행을 Kaggle로 옮기기 위한 데이터셋 준비.

공통 코호트(data/cohort_n110.txt, TCGA 110 + CPTAC 136)의 슬라이드만 골라
uni2native 특징(features_uni2native.pt, coords_uni2native.pt)과 slide_index CSV를 zip 두 개로 묶는다.
Kaggle은 업로드된 zip을 풀어서 보관하므로, 노트북에서는
  /kaggle/input/<dataset>/patches_tcga/{slide_index_task*.csv, tiles/<slide>/...}
처럼 보인다. 노트북이 이 폴더를 data/patches_tcga로 연결한다.

로컬에 uni2native 특징이 없는 TCGA 슬라이드 9장(전부 DX, 환자별 다른 슬라이드는 있음)은 건너뛰고
목록을 missing_uni2native.txt로 남긴다. HPC에서 받아 오면 별도 소형 데이터셋으로 올린다.

사용법: python -m scripts.compute.stage_kaggle_paad_features [--out data/kaggle_upload_uni2native_paad]
그 다음: python -m kaggle datasets create -p data/kaggle_upload_uni2native_paad
"""
import argparse
import glob
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILES = ("features_uni2native.pt", "coords_uni2native.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/kaggle_upload_uni2native_paad")
    ap.add_argument("--dataset-id", default="wonseukleejustinlee/path-vit-uni2native-paad")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cohort = set((ROOT / "data/cohort_n110.txt").read_text().split())
    missing = []
    for ds in ("tcga", "cptac"):
        src = ROOT / "data" / f"patches_{ds}"
        idx_files = sorted(glob.glob(str(src / "slide_index_task*.csv")))
        si = pd.concat([pd.read_csv(p) for p in idx_files])
        slides = si[si.case_id.isin(cohort)].slide_id.tolist()
        zpath = out / f"patches_{ds}.zip"
        n = 0
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for p in idx_files:
                z.write(p, arcname=f"patches_{ds}/{Path(p).name}")
            for sid in slides:
                d = src / "tiles" / sid
                if not all((d / f).exists() for f in FILES):
                    missing.append(f"{ds}\t{sid}")
                    continue
                for f in FILES:
                    z.write(d / f, arcname=f"patches_{ds}/tiles/{sid}/{f}")
                n += 1
        print(f"{ds}: {n} slides -> {zpath} ({zpath.stat().st_size / 1e9:.2f} GB)")
    (out / "missing_uni2native.txt").write_text("\n".join(missing) + "\n")
    print(f"missing {len(missing)} slides listed in missing_uni2native.txt")
    meta = {"title": "PATH-ViT uni2native PAAD cohort features", "id": args.dataset_id,
            "licenses": [{"name": "CC0-1.0"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
