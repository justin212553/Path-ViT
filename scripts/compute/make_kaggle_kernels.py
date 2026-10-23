"""
2026-10-22: PAAD QC 재실행용 Kaggle 커널 폴더를 파트별로 만든다(그리고 원하면 push).

  python -m scripts.compute.make_kaggle_kernels                 # 폴더만 생성 (.kaggle_kernels/qc-paad-part{k})
  python -m scripts.compute.make_kaggle_kernels --push 1        # 파트 1을 push = Kaggle에서 실행 시작
  python -m kaggle kernels status wonseukleejustinlee/path-vit-qc-paad-part1
  python -m kaggle kernels output wonseukleejustinlee/path-vit-qc-paad-part1 -p .kaggle_out/part1

push하면 Kaggle이 곧바로 GPU 세션을 시작하고 주간 GPU 한도(30시간)에서 시간이 빠진다.
결과 회수: python -m scripts.compute.collect_outputs .kaggle_out/part1  (예측 CSV를 .logs로 합침)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
USER = "wonseukleejustinlee"
# 원고 표에 들어가는 모델을 Kaggle에(옛 분석 번호). -noCNV 두 개는 로컬에서.
KAGGLE_MODELS = "M1 M2 M3 M4 M3SA M4SA M4NOMUT"
N_PARTS = 3
DATASETS = [f"{USER}/path-vit-uni2native-paad"]
EXTRA = f"{USER}/path-vit-uni2native-paad-extra"   # HPC에서 받아 온 9장(올린 뒤 --with-extra)


def build(part, workers, budget, with_extra):
    d = ROOT / ".kaggle_kernels" / f"qc-paad-part{part}"
    d.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "scripts" / "compute" / "kaggle_qc_paad_kernel.py").read_text(encoding="utf-8")
    src = (src.replace("__PART__", f"{part}/{N_PARTS}").replace("__MODELS__", KAGGLE_MODELS)
              .replace("__WORKERS__", str(workers)).replace("__BUDGET__", str(budget)))
    (d / "run.py").write_text(src, encoding="utf-8")
    meta = {
        "id": f"{USER}/path-vit-qc-paad-part{part}",
        "title": f"path-vit-qc-paad-part{part}",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": DATASETS + ([EXTRA] if with_extra else []),
        "competition_sources": [],
        "kernel_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--budget", type=float, default=11.0)
    ap.add_argument("--with-extra", action="store_true")
    ap.add_argument("--push", type=int, nargs="*", default=[])
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    args = ap.parse_args()
    for k in range(1, N_PARTS + 1):
        d = build(k, args.workers, args.budget, args.with_extra)
        print("built", d)
    for k in args.push:
        d = ROOT / ".kaggle_kernels" / f"qc-paad-part{k}"
        cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(d), "--accelerator", args.accelerator]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
