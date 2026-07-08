"""
k-means 군집 중심 사전 계산 스크립트 (LateFusionViT 전처리 1회용)

모든 노드의 features.pt (N_i, 2048)를 합산해 k-means를 실행하고
군집 중심을 cluster_centroids.pt (K, 2048)로 저장한다.

[실행 순서]
  1. python -m data.extract_features   # features.pt 사전 추출 (이미 완료된 경우 생략)
  2. python -m data.fit_clusters        # 군집 중심 계산 (본 스크립트)
  3. python train.py                    # LateFusionViT 학습

[저장 위치]
  <patches_root>/../cluster_centroids.pt  — patches_root 상위 디렉터리 (프로젝트 루트)

[K 선택 가이드]
  림프절 조직 유형 (종양 / 정상 림프구 / 간질 / 지방 / 혈관 / 괴사)을 기준으로
  K=8~16 권장. K가 너무 작으면 군집이 혼합되고, 너무 크면 히스토그램 벡터가 희소해진다.
  silhouette score로 K를 선택하려면 --eval-k 플래그를 사용한다.

사용법:
    python -m data.fit_clusters                    # 기본 K=10
    python -m data.fit_clusters --k 16             # K 지정
    python -m data.fit_clusters --eval-k 5 20      # K=5~20 실루엣 점수 비교 후 저장
    python -m data.fit_clusters --max-patches 5000 # 노드당 최대 샘플 수 제한 (속도)
"""
import argparse
import os
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
from data.patch_dataset import FEATURES_FILENAME
from utils import load_env, send_slack

CENTROIDS_DIR = "./data/cluster_centroids.pt"


def _load_all_features(patches_root: Path, max_patches_per_node: int) -> np.ndarray:
    """
    모든 노드의 features.pt를 읽어 하나의 numpy 배열로 합산한다.

    Args:
        max_patches_per_node: 노드당 최대 패치 수. 대용량 데이터에서 RAM 절감 및
                              속도 향상을 위해 랜덤 서브샘플링한다. 0이면 전체 사용.
    Returns:
        (N_total, 2048) float32 numpy array
    """
    node_dirs = sorted(d for d in patches_root.iterdir() if d.is_dir())
    chunks = []
    missing = 0

    for node_dir in node_dirs:
        feat_path = node_dir / FEATURES_FILENAME
        if not feat_path.exists():
            missing += 1
            continue
        feat = torch.load(feat_path, map_location="cpu").float().numpy()  # (N_i, 2048)

        if max_patches_per_node > 0 and len(feat) > max_patches_per_node:
            idx = np.random.choice(len(feat), max_patches_per_node, replace=False)
            feat = feat[idx]

        chunks.append(feat)

    if missing:
        print(f"  경고: features.pt 없는 노드 {missing}개 건너뜀 — data/extract_features.py 선실행 필요")

    all_features = np.concatenate(chunks, axis=0)  # (N_total, 2048)
    print(f"  로드 완료: {len(chunks)}개 노드 / 총 {len(all_features):,}개 패치")
    return all_features


def _fit_kmeans(features: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    MiniBatchKMeans로 k-means를 실행한다.
    MiniBatch를 사용하는 이유: 수백만 패치 규모에서 full k-means는 수 시간 소요,
    MiniBatch는 수 분 내 수렴하면서 품질 저하가 미미하다.

    Returns:
        centroids: (K, 2048) float32
    """
    print(f"  K={k} MiniBatchKMeans 실행 중...")
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=4096,
        n_init=3,
        max_iter=300,
        verbose=0,
    )
    km.fit(features)
    inertia = km.inertia_
    print(f"  완료 — inertia={inertia:.2f}")
    return km.cluster_centers_.astype(np.float32), km


def _eval_k_range(features: np.ndarray, k_min: int, k_max: int, seed: int) -> int:
    """
    k_min ~ k_max 범위에서 실루엣 점수를 계산해 최적 K를 반환한다.
    실루엣 점수 계산은 대용량에서 느리므로 최대 50,000개로 서브샘플링한다.
    """
    sample_size = min(50_000, len(features))
    idx = np.random.choice(len(features), sample_size, replace=False)
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

    print(f"\n  → 최적 K={best_k} (silhouette={best_score:.4f})")
    return best_k


def main():
    parser = argparse.ArgumentParser(description="k-means 군집 중심 사전 계산")
    parser.add_argument("--k", type=int, default=10,
                        help="군집 수 (기본 10)")
    parser.add_argument("--eval-k", type=int, nargs=2, metavar=("K_MIN", "K_MAX"),
                        help="K_MIN~K_MAX 범위의 실루엣 점수를 비교해 최적 K로 저장")
    parser.add_argument("--max-patches", type=int, default=0,
                        help="노드당 최대 패치 수 (0=전체, 속도/RAM 절감 시 지정)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_env()
    start_time = datetime.now()

    cfg = DataConfig()
    patches_root = Path(cfg.patches_root)
    out_path = _ROOT / CENTROIDS_DIR

    np.random.seed(args.seed)

    print(f"[1/3] features.pt 로드: {patches_root}")
    features = _load_all_features(patches_root, args.max_patches)

    if args.eval_k:
        k_min, k_max = args.eval_k
        print(f"\n[2/3] K 범위 탐색: {k_min}~{k_max}")
        best_k = _eval_k_range(features, k_min, k_max, args.seed)
        k = best_k
    else:
        k = args.k

    print(f"\n[2/3] K={k} 최종 k-means 실행")
    centroids, _ = _fit_kmeans(features, k, args.seed)

    print(f"\n[3/3] 저장: {out_path}")
    torch.save(torch.from_numpy(centroids), out_path)
    print(f"  cluster_centroids.pt 저장 완료 — shape: {centroids.shape}")
    print(
        f"\n다음 단계: train.py 실행 시 centroids가 자동으로 로드됩니다.\n"
        f"  python train.py"
    )

    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    send_slack(
        f":white_check_mark: *fit_clusters 완료*\n"
        f"> K={k} | 패치 수: {len(features):,} | 저장: `{out_path.name}`\n"
        f"> 소요 시간: {h}h {m}m {s}s"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        send_slack(f":x: *fit_clusters 에러*\n```{type(e).__name__}: {e}```")
        raise
