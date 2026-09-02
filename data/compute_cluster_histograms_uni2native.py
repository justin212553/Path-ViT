"""
HDP(Human Doctor Prognosis) 모델의 feature 전처리 — data/fit_clusters_uni2native.py가 계산한
K개 군집 중심에 대해, 환자별로 "patch가 각 군집에 얼마나 분포하는가"(K차원 히스토그램,
비율 합=1)를 계산해 CSV로 저장한다.

[배경, 2026-09-01] 클러스터 대표 patch를 눈으로 봐도(scripts/extract_cluster_exemplars.py)
어느 군집이 "종양"인지 사람이 확신 있게 판정할 근거가 TCGA/CPTAC 어디에도 없다(clinical.tsv,
pathology_detail.tsv 전부 PAAD에서 PNI/TIL/종양비율 필드가 100% 결측 — 확인 완료). 그래서
군집에 의미를 붙이려 하지 않고, 각 군집의 "비율"을 그 자체로 하나의 저차원 구조화 feature로
써서 152개 생존 라벨이 어느 군집 비율이 hazard와 상관있는지 직접 결정하게 한다 —
`data/fit_clusters.py`(구버전 resnet50, LateFusionViT `--fusion`)가 이미 시도했던 방식을
uni2native(공식 스펙) 피처로 다시 하는 것.

환자가 슬라이드를 여러 장 가지면(TCGA에 흔함) 슬라이드별 raw count를 합산한 뒤 정규화한다 —
슬라이드 하나짜리 환자와 동일한 정규화 스케일(비율 합=1)을 보장하기 위함.

사용법:
    python -m data.compute_cluster_histograms_uni2native
    python -m data.compute_cluster_histograms_uni2native --datasets tcga
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FEATURES_ROOT = _ROOT / "data" / "uni2h_official_features"
CENTROIDS_PATH = _ROOT / "data" / "cluster_centroids_uni2native.pt"
OUT_PATHS = {
    "tcga": _ROOT / "data" / "cluster_hist_uni2native_tcga.csv",
    "cptac": _ROOT / "data" / "cluster_hist_uni2native_cptac.csv",
}
# h5 파일명(stem) -> case_id 규칙. clinical_{tcga,cptac}.csv의 case_id 포맷과 직접 대조 확인함
# (2026-09-01): TCGA "TCGA-2J-AAB1-01Z-00-DX1...." -> "TCGA-2J-AAB1"(앞 3토큰),
# CPTAC "C3L-00017-21" -> "C3L-00017"(앞 2토큰).
CASE_ID_TOKENS = {"tcga": 3, "cptac": 2}


def _case_id_from_stem(stem: str, dataset: str) -> str:
    parts = stem.split("-")
    n = CASE_ID_TOKENS[dataset]
    return "-".join(parts[:n])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default="tcga,cptac")
    args = parser.parse_args()

    centroids = torch.load(CENTROIDS_PATH).numpy()  # (K, 1536)
    k = centroids.shape[0]
    print(f"K={k} 군집 중심 로드: {CENTROIDS_PATH}")

    for ds in args.datasets.split(","):
        ds_dir = FEATURES_ROOT / ds
        h5_paths = sorted(ds_dir.glob("*.h5"))
        print(f"\n=== {ds}: {len(h5_paths)}개 슬라이드 ===")

        case_counts: dict[str, np.ndarray] = {}
        for h5_path in h5_paths:
            case_id = _case_id_from_stem(h5_path.stem, ds)
            with h5py.File(h5_path, "r") as f:
                feat = f["features"][0]  # (N, 1536)
            # (N, K) 거리 -> 최근접 군집. 슬라이드 하나씩 처리라 메모리 부담 적음(N~1만 x K~10 x 1536).
            d = np.linalg.norm(feat[:, None, :] - centroids[None, :, :], axis=-1)  # (N, K)
            assign = d.argmin(axis=1)  # (N,)
            counts = np.bincount(assign, minlength=k).astype(np.float64)
            if case_id in case_counts:
                case_counts[case_id] += counts
            else:
                case_counts[case_id] = counts

        rows = []
        for case_id, counts in sorted(case_counts.items()):
            total = counts.sum()
            frac = counts / total if total > 0 else counts
            rows.append({"case_id": case_id, **{f"hist_{i}": frac[i] for i in range(k)},
                         "n_patches": int(total)})
        df = pd.DataFrame(rows)
        out_path = OUT_PATHS[ds]
        df.to_csv(out_path, index=False)
        print(f"  {len(df)}명 환자 -> {out_path}")
        print(f"  patch 수 분포: min={df['n_patches'].min()}, median={df['n_patches'].median():.0f}, "
              f"max={df['n_patches'].max()}")


if __name__ == "__main__":
    main()
