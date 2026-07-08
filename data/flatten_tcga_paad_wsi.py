"""
data/tcga_paad_wsi 하위 폴더들에 흩어져 있는 .svs 파일들을 전부
data/tcga_paad_wsi 바로 아래로 옮기고, 비워진 하위 폴더는 삭제한다.

사용법:
    python -m data.flatten_tcga_paad_wsi
"""
import shutil
from pathlib import Path

ROOT = Path("./data/tcga_paad_wsi")


def flatten_svs(root: Path = ROOT) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"{root} 폴더가 존재하지 않습니다.")

    subdirs = [p for p in root.iterdir() if p.is_dir()]

    for subdir in subdirs:
        for svs_path in subdir.rglob("*.svs"):
            dest = root / svs_path.name
            if dest.exists():
                print(f"[skip] 이미 존재하는 파일명이라 건너뜀: {dest}")
                continue
            shutil.move(str(svs_path), str(dest))
            print(f"[move] {svs_path} -> {dest}")

        shutil.rmtree(subdir)
        print(f"[rmdir] {subdir}")


if __name__ == "__main__":
    flatten_svs()
