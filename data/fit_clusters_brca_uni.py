"""
data/fit_clusters_uni2native.py의 BRCA 버전 — PAAD의 클러스터풀(models/vit_pma.py cluster_pool,
Nystrom+ABMIL을 완전히 우회하고 raw feature 공간 unsupervised 군집으로 N개 패치를 K개 대표
토큰으로 요약한 뒤 RNA co-attention에 바로 넘기는 구조, 2026-09-05 PAAD에서 M7과 통계적으로
동등한 수준까지 WSI의 "해로움"을 없앤 것으로 확인됨)을 BRCA(N=1058, PAAD의 ~10배 — WSI 기여의
통계적 유의성을 검정할 힘이 훨씬 큰 코호트)에 이식하기 위한 1단계.

BRCA는 backbone="uni"(UNI v1, 1024차원 — PAAD의 uni2native/uni2와 다른 feature 공간이라
그쪽 centroids를 재사용할 수 없다)이고, 파일 레이아웃도 다르다(h5 한 파일/slide가 아니라
data/patches_tcga_brca/tiles/{slide_id}/features_uni.pt, scripts/brca_common.py 참조).

사용법(HPC에서 실행 — BRCA feature가 로컬엔 없고 HPC에만 있음, brca_for_hpc.zip 참조):
    python -m data.fit_clusters_brca_uni --eval-k 6 16   # silhouette로 K 탐색 후 그 K로 최종 적합
    python -m data.fit_clusters_brca_uni --k 11           # K 고정
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MANIFEST_PATH = _ROOT / "data" / "brca_slide_manifest.csv"
TILES_ROOT = _ROOT / "data" / "patches_tcga_brca" / "tiles"
OUT_PATH = _ROOT / "data" / "cluster_centroids_brca_uni.pt"


def _load_all_features(max_patches_per_slide: int, seed: int) -> np.ndarray:
    manifest = pd.read_csv(MANIFEST_PATH)
    rng = np.random.default_rng(seed)
    chunks = []
    n_slides = 0
    for slide_id in manifest["slide_id"]:
        feat_path = TILES_ROOT / slide_id / "features_uni.pt"
        if not feat_path.exists():
            continue
        feat = torch.load(feat_path, weights_only=True).numpy().astype(np.float32)  # (N, 1024)
        n = feat.shape[0]
        if max_patches_per_slide > 0 and n > max_patches_per_slide:
            idx = rng.choice(n, max_patches_per_slide, replace=False)
            idx.sort()
            feat = feat[idx]
        chunks.append(feat)
        n_slides += 1
    features = np.concatenate(chunks, axis=0)
    print(f"  로드 완료: {n_slides}개 슬라이드 / 총 {len(features):,}개 patch (슬라이드당 최대 {max_patches_per_slide or '전체'})")
    return features


def _fit_kmeans(features: np.ndarray, k: int, seed: int):
    print(f"  K={k} MiniBatchKMeans 실행 중...")
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3, max_iter=300, verbose=0)
    km.fit(features)
    print(f"  완료 — inertia={km.inertia_:.2f}")
    return km.cluster_centers_.astype(np.float32), km


def _eval_k_range(features: np.ndarray, k_min: int, k_max: int, seed: int) -> int:
    sample_size = min(50_000, len(features))
    idx = np.random.default_rng(seed).choice(len(features), sample_size, replace=False)
    sample = features[idx]
    best_k, best_score = k_min, -1.0
    print(f"\n  K 범위 {k_min}~{k_max} 실루엣 점수 평가 (샘플 {sample_size:,}개):")
    for k in range(k_min, k_max + 1):
        _, km = _fit_kmeans(features, k, seed)
        labels = km.predict(sample)
        score = silhouette_score(sample, labels, sample_size=min(10_000, sample_size))
        print(f"    K={k:3d}  silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k
    print(f"\n  -> 최적 K={best_k} (silhouette={best_score:.4f})")
    return best_k


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--eval-k", type=int, nargs=2, metavar=("K_MIN", "K_MAX"))
    parser.add_argument("--max-patches-per-slide", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-path", type=str, default=None)
    args = parser.parse_args()
    out_path = Path(args.out_path) if args.out_path else OUT_PATH

    start = datetime.now()
    print("[1/3] BRCA uni feature 로드")
    features = _load_all_features(args.max_patches_per_slide, args.seed)

    if args.eval_k:
        k_min, k_max = args.eval_k
        print(f"\n[2/3] K 범위 탐색: {k_min}~{k_max}")
        k = _eval_k_range(features, k_min, k_max, args.seed)
    else:
        k = args.k

    print(f"\n[2/3] K={k} 최종 k-means 실행")
    centroids, km = _fit_kmeans(features, k, args.seed)

    print(f"\n[3/3] 저장: {out_path}")
    torch.save(torch.from_numpy(centroids), out_path)

    sizes = np.bincount(km.labels_, minlength=k)
    print(f"  군집별 patch 수(샘플 {len(km.labels_):,}개 기준): {sizes.tolist()}")
    print(f"  군집별 비율: {(sizes / sizes.sum() * 100).round(1).tolist()}%")

    elapsed = datetime.now() - start
    print(f"\n완료 — 소요 시간: {elapsed}")


if __name__ == "__main__":
    main()
