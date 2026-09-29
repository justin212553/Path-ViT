"""
ClusterPool(models/vit_pma.py) K=11 기본 centroids를 만든 것과 같은 스크립트 — TCGA-PAAD/
CPTAC-PDAC의 uni2native 패치 feature(우리 raw WSI를 우리 파이프라인으로 재타일링한 UNI2-h
feature, scripts/reconcile_uni2native_features.py 산출물)를 라벨 없이 k-means로 군집화한다.

[2026-09-28 수정] 이 스크립트는 원래 MahmoodLab의 gated HuggingFace dataset에서 받은 "공식"
UNI2-h feature(data/uni2h_official_features/{tcga,cptac}/*.h5, scripts/
download_uni2h_official_features.py 산출물)를 읽고 있었다 — "우리 자체 추출 파이프라인이
UNI2-h 공식 학습 스펙과 4배 이상 어긋난다"는 가설을 검증하려던 2026-08-12 대조실험용 브랜치의
잔재다. 그 실험은 이미 끝났고 해당 h5 디렉터리도 몇 주 전에 지워졌는데(용량 정리), 이 스크립트의
기본 로딩 경로는 그대로 남아 있어 K 민감도 재적합(--k 5/--k 20) 때 h5 파일을 하나도 못 찾고
`ValueError: need at least one array to concatenate`로 죽었다. 실제로 지금 학습(train.py
--backbone uni2native)이 쓰는 feature는 h5가 아니라, data/dataset.py::FEATURES_FILENAME_BY_
BACKBONE["uni2native"] = "features_uni2native.pt"이 가리키는 슬라이드별 .pt 파일
(patches_root_{tcga,cptac}/tiles/<slide_id>/features_uni2native.pt, config.py 기본값
data/patches_tcga·data/patches_cptac) — data/fit_clusters.py(구버전 resnet50 2048차원)와
정확히 같은 저장 구조다. 이 스크립트도 그 구조를 읽도록 고쳤다. 기존 K=11 centroids
(data/cluster_centroids_uni2native.pt)는 이 수정과 무관하게 그대로 유효하다(어떤 경로로
만들어졌든 재사용 가능한 결과물 자체는 이미 저장돼 있음) — 이번에 새로 재적합하는 K=5/K=20만
이 수정된 경로를 탄다.

사용법:
    python -m data.fit_clusters_uni2native                      # 기본: tcga+cptac 합산, K=10
    python -m data.fit_clusters_uni2native --k 12
    python -m data.fit_clusters_uni2native --eval-k 6 16         # silhouette로 K 탐색
    python -m data.fit_clusters_uni2native --max-patches-per-slide 3000
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DataConfig
from data.dataset import PATCHES_ROOT_ATTRS
from data.patch_utils import FEATURES_UNI2NATIVE_FILENAME

OUT_PATH = _ROOT / "data" / "cluster_centroids_uni2native.pt"
OUT_META_PATH = _ROOT / "data" / "cluster_centroids_uni2native_meta.pt"


def _load_all_features(datasets: list[str], max_patches_per_slide: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    """
    각 슬라이드 디렉터리의 features_uni2native.pt(N, 1536)를 읽어 슬라이드당 최대
    max_patches_per_slide개로 서브샘플링한 뒤 합친다(data/fit_clusters.py와 동일 구조,
    파일명/backbone만 uni2native).

    Returns:
        features: (N_total, 1536) float32
        slide_meta: 각 patch가 어느 slide/coord에서 왔는지(추후 exemplar 추출용) —
            [{"dataset":..., "slide_path":..., "coord_idx": np.ndarray}, ...]
    """
    rng = np.random.default_rng(seed)
    cfg = DataConfig()
    chunks = []
    slide_meta = []
    missing = 0
    n_slides = 0

    for ds in datasets:
        patches_root = Path(getattr(cfg, PATCHES_ROOT_ATTRS[ds]))
        tiles_root = patches_root / "tiles"
        slide_dirs = sorted(d for d in tiles_root.iterdir() if d.is_dir())
        for slide_dir in slide_dirs:
            feat_path = slide_dir / FEATURES_UNI2NATIVE_FILENAME
            if not feat_path.exists():
                missing += 1
                continue
            feat = torch.load(feat_path, map_location="cpu").float().numpy()  # (N, 1536)
            n = feat.shape[0]
            if max_patches_per_slide > 0 and n > max_patches_per_slide:
                idx = rng.choice(n, max_patches_per_slide, replace=False)
                idx.sort()
            else:
                idx = np.arange(n)
            chunks.append(feat[idx].astype(np.float32))
            slide_meta.append({"dataset": ds, "slide_path": str(slide_dir), "coord_idx": idx})
            n_slides += 1

    if missing:
        print(f"  경고: {FEATURES_UNI2NATIVE_FILENAME} 없는 슬라이드 {missing}개 건너뜀 — "
              f"scripts/reconcile_uni2native_features.py 선실행 필요")
    if not chunks:
        raise FileNotFoundError(
            f"{FEATURES_UNI2NATIVE_FILENAME}를 가진 슬라이드를 하나도 못 찾았습니다 "
            f"(datasets={datasets}). patches_root_{{tcga,cptac}}(config.py) 경로와 "
            f"scripts/reconcile_uni2native_features.py 실행 여부를 확인하세요."
        )

    features = np.concatenate(chunks, axis=0)
    print(f"  로드 완료: {n_slides}개 슬라이드 / 총 {len(features):,}개 patch (슬라이드당 최대 {max_patches_per_slide or '전체'})")
    return features, slide_meta


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
    parser.add_argument("--datasets", type=str, default="tcga,cptac")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eval-k", type=int, nargs=2, metavar=("K_MIN", "K_MAX"))
    parser.add_argument("--max-patches-per-slide", type=int, default=3000,
                         help="슬라이드당 최대 샘플 patch 수(0=전체, 슬라이드당 평균 ~9600개라 "
                              "기본값 3000이면 전체의 ~31%%만 써도 k-means엔 충분).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-path", type=str, default=None,
                         help="2026-09-05: 기본(None)이면 OUT_PATH(data/cluster_centroids_uni2native.pt) "
                              "덮어씀 — 재적합 버전끼리 비교할 때 다른 경로로 저장하기 위함.")
    args = parser.parse_args()
    out_path = Path(args.out_path) if args.out_path else OUT_PATH

    datasets = args.datasets.split(",")
    start = datetime.now()

    print(f"[1/3] uni2native feature 로드: {datasets}")
    features, slide_meta = _load_all_features(datasets, args.max_patches_per_slide, args.seed)

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

    # 군집별 patch 비율(대략적인 크기 감) — exemplar 추출 전에도 "이 군집이 흔한지 희귀한지" 참고용
    labels_all = km.labels_
    sizes = np.bincount(labels_all, minlength=k)
    print(f"  군집별 patch 수(샘플 {len(labels_all):,}개 기준): {sizes.tolist()}")
    print(f"  군집별 비율: {(sizes / sizes.sum() * 100).round(1).tolist()}%")

    meta_path = out_path.with_name(out_path.stem + "_meta.pt")
    torch.save({"k": k, "sizes": sizes.tolist(), "n_slides": len(slide_meta), "datasets": datasets}, meta_path)

    elapsed = datetime.now() - start
    print(f"\n완료 — 소요 시간: {elapsed}")
    print("다음 단계: scripts/extract_cluster_exemplars.py로 군집별 대표 patch 이미지를 뽑아 "
          "눈으로 보고 '이 군집=종양처럼 보인다'를 판정.")


if __name__ == "__main__":
    main()
