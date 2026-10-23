"""
2026-10-22: HPC에서 받은 pathvit_missing9.tar(로컬에 없던 TCGA DX 9장)를 로컬 data/patches_tcga/tiles에 풀고,
Kaggle 보조 데이터셋(path-vit-uni2native-paad-extra)으로 올린다. 커널은 extra_tcga/tiles/<slide>/를 본 트리에 합친다.

  python -m scripts.compute.stage_kaggle_extra pathvit_missing9.tar [--no-upload]
그 뒤 커널을 --with-extra로 다시 만들어 push:
  python -m scripts.compute.make_kaggle_kernels --with-extra --push 1
"""
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
USER = "wonseukleejustinlee"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tar", type=Path)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    with tarfile.open(args.tar) as t:
        members = [m for m in t.getmembers() if m.isfile() and m.name.startswith("data/patches_tcga/tiles/")]
        t.extractall(ROOT, members=members)
    print(f"local: {len(members)} files -> data/patches_tcga/tiles")
    out = ROOT / "data" / "kaggle_upload_uni2native_paad_extra"
    if out.exists():
        shutil.rmtree(out)
    for m in members:
        rel = Path(m.name).relative_to("data/patches_tcga")
        dst = out / "extra_tcga" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / m.name, dst)
    meta = {"title": "PATH-ViT uni2native PAAD extra slides", "id": f"{USER}/path-vit-uni2native-paad-extra",
            "licenses": [{"name": "CC0-1.0"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    if not args.no_upload:
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "create", "-p", str(out.resolve()), "--dir-mode", "zip"], check=True)


if __name__ == "__main__":
    main()
