"""
HDP(Human Doctor Prognosis) 모델의 1단계 — UNI2-h 공식 스펙(20x, ~0.5um/px,
`data/uni2h_official_features/{tcga,cptac}/*.h5`) patch feature를 라벨 없이 k-means로
군집화한다. `data/fit_clusters.py`(구버전 resnet50 2048차원, LateFusionViT 전처리용)와 같은
아이디어를 UNI2-h 1536차원 feature에 맞게 새로 짠 것 — 파일 형식이 .pt(per-slide 디렉토리)가
아니라 .h5(slide당 한 파일, MahmoodLab 공식 배포 그대로)라 로딩 경로가 다르다.

[배경] 2026-09-01 WSI branch 재설계 논의 — "종양 patch 어디 있는지"를 새 라벨/외부
데이터셋 없이(사용자 결정) 알아내려면, frozen UNI2-h feature를 비지도 군집화해서 군집 대표
patch 몇 장을 사람이 눈으로 보고 "이 군집은 종양처럼 생겼다"를 사후 판정하는 방법뿐이다.
이 스크립트는 그 첫 단계(군집 중심 계산)만 한다 — 대표 patch 이미지 추출/시각화는 별도
스크립트(scripts/extract_cluster_exemplars.py, 다음 단계)에서 한다.

[해상도 관련 중요 사실] 이 프로젝트 자체 추출 파이프라인(--backbone uni2, 1024px@1.0MPP
->512/224 리사이즈, 실효 2~4.57um/px)은 UNI2-h 공식 학습 스펙(256px@20x, ~0.5um/px)과
4배 이상 어긋난다(scripts/download_uni2h_official_features.py 상단 docstring, 2026-08-12
확인) — 개별 핵조차 구별 안 되는 해상도다. 이 스크립트가 쓰는 uni2native(공식) feature는
그 문제가 없다 — "종양 영역처럼 보이는 patch"를 군집이 실제로 분리해낼 가능성이 우리 자체
파이프라인보다 훨씬 높다.

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

import h5py
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FEATURES_ROOT = _ROOT / "data" / "uni2h_official_features"
OUT_PATH = _ROOT / "data" / "cluster_centroids_uni2native.pt"
OUT_META_PATH = _ROOT / "data" / "cluster_centroids_uni2native_meta.pt"


def _load_all_features(datasets: list[str], max_patches_per_slide: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    """
    각 h5(slide 하나)에서 features(1, N, 1536)를 읽어 슬라이드당 최대
    max_patches_per_slide개로 서브샘플링한 뒤 합친다.

    Returns:
        features: (N_total, 1536) float32
        slide_meta: 각 patch가 어느 slide/coord에서 왔는지(추후 exemplar 추출용) —
            [{"dataset":..., "case_id":..., "slide_path":..., "coord_idx": np.ndarray}, ...]
    """
    rng = np.random.default_rng(seed)
    chunks = []
    slide_meta = []
    n_slides = 0

    for ds in datasets:
        ds_dir = FEATURES_ROOT / ds
        h5_paths = sorted(ds_dir.glob("*.h5"))
        for p in h5_paths:
            with h5py.File(p, "r") as f:
                feat = f["features"][0]  # (N, 1536)
            n = feat.shape[0]
            if max_patches_per_slide > 0 and n > max_patches_per_slide:
                idx = rng.choice(n, max_patches_per_slide, replace=False)
                idx.sort()
            else:
                idx = np.arange(n)
            chunks.append(feat[idx].astype(np.float32))
            slide_meta.append({"dataset": ds, "slide_path": str(p), "coord_idx": idx})
            n_slides += 1

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
