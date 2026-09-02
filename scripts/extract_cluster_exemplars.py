"""
HDP(Human Doctor Prognosis) 모델 2단계 — data/fit_clusters_uni2native.py가 계산한 K개 군집
중심에 대해, 실제로 그 군집에 가장 가까운 patch들의 원본 이미지를 뽑아 사람이 눈으로 보고
"이 군집은 종양처럼 생겼다/기질이다/..." 를 사후 판정할 수 있게 한다 — 새 라벨/외부
데이터셋 없이, 순수 비지도 군집화 결과를 사람이 해석하는 방식(2026-09-01 사용자 결정).

patch 좌표(coords, level-0 픽셀)는 512x512 stride인데 UNI2-h 공식 feature는 256x256@20x
스펙이다 — 즉 level-0(40x 네이티브)에서 512x512를 읽어 256x256으로 다운샘플한 게 실제로
모델이 본 patch다(128um x 128um 물리 영역, 512*0.25um = 256*0.5um). 이 스크립트도 동일하게
읽는다.

사용법:
    python -m scripts.extract_cluster_exemplars                    # 군집당 9장, 기본 경로
    python -m scripts.extract_cluster_exemplars --n-per-cluster 12
    python -m scripts.extract_cluster_exemplars --max-patches-per-slide 1000  # fit 때와 맞추기
"""
import argparse
import heapq
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FEATURES_ROOT = _ROOT / "data" / "uni2h_official_features"
CENTROIDS_PATH = _ROOT / "data" / "cluster_centroids_uni2native.pt"
WSI_ROOTS = {"tcga": _ROOT / "data" / "tcga_paad_wsi", "cptac": _ROOT / "data" / "cptac_pda_wsi"}
READ_SIZE_LEVEL0 = 512  # coords stride와 동일 — 이 크기로 읽어 256으로 리사이즈하면 공식 patch와 일치
DISPLAY_SIZE = 256


def _find_wsi_path(dataset: str, slide_id: str) -> Path | None:
    root = WSI_ROOTS[dataset]
    cand = root / f"{slide_id}.svs"
    if cand.exists():
        return cand
    matches = list(root.glob(f"{slide_id}*.svs"))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default="tcga,cptac")
    parser.add_argument("--n-per-cluster", type=int, default=9)
    parser.add_argument("--max-patches-per-slide", type=int, default=1000,
                         help="fit_clusters_uni2native.py와 같은 값으로 맞추면 재현성 있음(필수는 아님).")
    parser.add_argument("--out-dir", type=str,
                         default=str(Path.home() / "AppData/Local/Temp/claude" /
                                     "d--wonse-Documents-Job-urban-datalab-PATH-ViT" /
                                     "31b48df2-b2a3-4986-995d-b8fef7cad784/scratchpad/cluster_exemplars"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    centroids = torch.load(CENTROIDS_PATH).numpy()  # (K, 1536)
    k = centroids.shape[0]
    print(f"K={k} 군집, 군집당 상위 {args.n_per_cluster}개 exemplar 추출")

    # 군집별 top-N 후보: (음의 거리, 순번, dataset, h5_path, patch_row_idx, coord) 최소힙으로 유지
    heaps: list[list] = [[] for _ in range(k)]
    counter = 0
    rng = np.random.default_rng(args.seed)

    datasets = args.datasets.split(",")
    for ds in datasets:
        ds_dir = FEATURES_ROOT / ds
        h5_paths = sorted(ds_dir.glob("*.h5"))
        for h5_path in h5_paths:
            with h5py.File(h5_path, "r") as f:
                feat = f["features"][0]  # (N, 1536)
                coords = f["coords"][0]  # (N, 2)
            n = feat.shape[0]
            if args.max_patches_per_slide > 0 and n > args.max_patches_per_slide:
                idx = rng.choice(n, args.max_patches_per_slide, replace=False)
            else:
                idx = np.arange(n)
            sub_feat = feat[idx]
            sub_coords = coords[idx]

            # (n_sub, K) 거리 — 슬라이드 하나씩이라 메모리 부담 적음
            d = np.linalg.norm(sub_feat[:, None, :] - centroids[None, :, :], axis=-1)  # (n_sub, K)
            for ci in range(k):
                for row in range(len(idx)):
                    dist = float(d[row, ci])
                    counter += 1
                    entry = (-dist, counter, ds, str(h5_path), sub_coords[row].tolist())
                    if len(heaps[ci]) < args.n_per_cluster:
                        heapq.heappush(heaps[ci], entry)
                    elif -dist > heaps[ci][0][0]:
                        heapq.heapreplace(heaps[ci], entry)
        print(f"  {ds}: {len(h5_paths)}개 슬라이드 처리 완료")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wsi_cache: dict[str, openslide.OpenSlide] = {}

    for ci in range(k):
        entries = sorted(heaps[ci], key=lambda e: -e[0])  # 거리 오름차순(가까운 순)
        imgs = []
        for (_, _, ds, h5_path, coord) in entries:
            slide_id = Path(h5_path).stem
            if h5_path not in wsi_cache:
                wsi_path = _find_wsi_path(ds, slide_id)
                if wsi_path is None:
                    print(f"    [경고] WSI 원본을 못 찾음: {ds}/{slide_id}")
                    wsi_cache[h5_path] = None
                else:
                    wsi_cache[h5_path] = openslide.OpenSlide(str(wsi_path))
            slide = wsi_cache[h5_path]
            if slide is None:
                continue
            x, y = coord
            region = slide.read_region((int(x), int(y)), 0, (READ_SIZE_LEVEL0, READ_SIZE_LEVEL0)).convert("RGB")
            region = region.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
            imgs.append(region)

        if not imgs:
            print(f"  cluster {ci}: 이미지 없음(원본 WSI 못 찾음)")
            continue

        cols = min(3, len(imgs))
        rows = (len(imgs) + cols - 1) // cols
        grid = Image.new("RGB", (cols * DISPLAY_SIZE, rows * DISPLAY_SIZE), "white")
        for i, img in enumerate(imgs):
            r, c = divmod(i, cols)
            grid.paste(img, (c * DISPLAY_SIZE, r * DISPLAY_SIZE))
        out_path = out_dir / f"cluster_{ci:02d}.png"
        grid.save(out_path)
        print(f"  cluster {ci}: {len(imgs)}장 -> {out_path}")

    for slide in wsi_cache.values():
        if slide is not None:
            slide.close()

    print(f"\n완료 — {out_dir} 에서 cluster_00.png ~ cluster_{k-1:02d}.png 확인.")


if __name__ == "__main__":
    main()
