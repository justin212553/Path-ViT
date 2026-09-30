"""
Fold-safe 전처리 — 모든 정규화 통계와 cluster centroid를 그 seed x fold의 train split 환자만으로
계산한다(2026-09-29, 리뷰 지적: 코호트 전체로 한 번 계산한 통계를 모든 fold에 재사용하면 val/test/
external 정보가 전처리에 섞인다).

train.py / train_light.py가 train/val/test/external dataset을 다 만든 직후, 모델을 만들기 전에
호출한다:
    apply_fold_safe_molecular(train_ds, val_ds, test_ds, external_ds)   # RNA/CNV
    stats = fold_safe_clinical_stats(train_ds.cases, clinical_paths)    # age/stage/margin/mutation
    path  = fit_fold_safe_centroids(train_ds, k=11)                      # ClusterPool centroids

external은 external 자신의 통계가 아니라 학습 fold 통계로 정규화한다(strict external validation).
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.clinical_encoder import stage_stats_from_df, margin_stats_from_df, mutation_stats_from_df

_ROOT = Path(__file__).resolve().parent.parent
CENTROID_CACHE_DIR = _ROOT / "data" / "centroids_fs"


def apply_fold_safe_molecular(train_ds, *other_dss) -> None:
    """train_ds(molecular_raw=True)의 case들로 RNA/CNV 통계를 내 train 포함 모든 dataset에 적용."""
    stats = train_ds.molecular_norm_stats()
    for ds in (train_ds, *other_dss):
        if ds is not None:
            ds.apply_molecular_norm(stats)
    print(f"[fold-safe] RNA/CNV 정규화 통계: train {len(train_ds.cases)}명 기준으로 계산, "
          f"{1 + sum(d is not None for d in other_dss)}개 dataset에 적용")


def fold_safe_clinical_stats(train_case_ids, clinical_paths: list) -> dict:
    """train 환자만으로 age mean/std와 stage/margin/mutation 통계를 계산한다. 기존 train.py가
    코호트 전체 CSV로 계산하던 것과 같은 함수(ddof=0)를 train 행에만 적용한다."""
    df = pd.concat([pd.read_csv(p) for p in clinical_paths], ignore_index=True)
    df = df[df["case_id"].isin(set(train_case_ids))]
    ages = df["age_years"].astype(float)
    out = {"age": (float(ages.mean()), float(ages.std(ddof=0)))}
    for key, fn, needed in [("stage", stage_stats_from_df, "ajcc_t"),
                            ("margin", margin_stats_from_df, "residual_disease"),
                            ("mutation", mutation_stats_from_df, "KRAS_mut")]:
        out[key] = fn(df) if needed in df.columns else None
    print(f"[fold-safe] clinical 정규화 통계: train {len(df)}명 기준 (age mean={out['age'][0]:.2f})")
    return out


def fit_centroids_from_files(feature_files: list, case_ids, backbone: str, k: int = 11,
                             max_patches_per_slide: int = 3000, kmeans_seed: int = 42,
                             cache_dir: Path = CENTROID_CACHE_DIR) -> str:
    """주어진 슬라이드 feature 파일들(= train split 슬라이드)로만 k-means centroid를 적합해 저장한다.

    캐시 키는 (train 환자 집합, backbone, k, 샘플링 설정)의 해시 — 같은 seed x fold를 쓰는 모델
    (M1~M4, SA 변형 등)은 train 집합이 같으므로 한 번만 적합된다. HPC에서 여러 array task가 동시에
    같은 파일을 쓸 수 있어 임시 파일에 쓴 뒤 os.replace로 원자적으로 교체한다. 파일명은 짧게 둔다
    (Windows 경로 길이). 호출부는 이 경로를 모델에만 넘기고 태그에는 넣지 않는다(fold마다 달라짐)."""
    from sklearn.cluster import MiniBatchKMeans

    key_src = f"{backbone}|k{k}|mp{max_patches_per_slide}|s{kmeans_seed}|" + ",".join(sorted(case_ids))
    key = hashlib.sha1(key_src.encode()).hexdigest()[:10]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"c{k}_{key}.pt"
    if path.exists():
        print(f"[fold-safe] centroid 캐시 재사용: {path.name}")
        return str(path)

    rng = np.random.default_rng(kmeans_seed)
    chunks = []
    for fp in feature_files:
        feat = torch.load(fp, map_location="cpu", weights_only=True).float().numpy()
        n = feat.shape[0]
        idx = np.sort(rng.choice(n, max_patches_per_slide, replace=False)) if n > max_patches_per_slide else np.arange(n)
        chunks.append(feat[idx])
    feats = np.concatenate(chunks, axis=0).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=k, random_state=kmeans_seed, batch_size=4096, n_init=3, max_iter=300)
    km.fit(feats)

    tmp = path.with_suffix(f".tmp{os.getpid()}")
    torch.save(torch.from_numpy(km.cluster_centers_.astype(np.float32)), tmp)
    os.replace(tmp, path)
    print(f"[fold-safe] centroid 적합: train {len(set(case_ids))}명 / {len(chunks)}슬라이드 / "
          f"{len(feats):,} patch, K={k} -> {path.name}")
    return str(path)


def fit_fold_safe_centroids(train_ds, k: int = 11, **kw) -> str:
    """WSISurvivalDataset(train split)용 — 모델이 실제로 쓰는 train 슬라이드로만 centroid 적합."""
    files = [train_ds.roots[r["dataset"]] / "tiles" / r["slide_id"] / train_ds.features_filename
             for _, r in train_ds.items.iterrows()]
    return fit_centroids_from_files(files, train_ds.cases, train_ds.feature_backbone, k=k, **kw)
