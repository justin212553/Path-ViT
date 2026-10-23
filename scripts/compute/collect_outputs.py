"""
2026-10-22: Kaggle·Vast.ai에서 받아 온 출력 폴더의 예측 CSV와 done 표시를 로컬 .logs로 합친다.
이미 같은 이름의 파일이 있으면 덮어쓰지 않고 건너뛴다(--overwrite로 강제).

  python -m scripts.compute.collect_outputs .kaggle_out/part1 [.kaggle_out/part2 ...]
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUBS = ("kfold_preds", "external_preds", "compute_done", "compute_jobs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    for d in args.dirs:
        found = list(d.rglob(".logs"))
        if not found:
            print(f"{d}: .logs 폴더 없음")
            continue
        for logs in found:
            for sub in SUBS:
                src = logs / sub
                if not src.is_dir():
                    continue
                dst = ROOT / ".logs" / sub
                dst.mkdir(parents=True, exist_ok=True)
                n = skip = 0
                for p in src.iterdir():
                    q = dst / p.name
                    if q.exists() and not args.overwrite:
                        skip += 1
                        continue
                    shutil.copy2(p, q)
                    n += 1
                print(f"{logs}/{sub}: {n} copied, {skip} skipped")


if __name__ == "__main__":
    main()
