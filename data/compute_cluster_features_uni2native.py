"""
HDP(Human Doctor Prognosis) 모델의 feature 전처리 v2 — data/compute_cluster_histograms_uni2native.py
(비율만, K차원)를 확장해, 군집별로 4가지 라벨-프리 통계를 계산한다(4*K차원):

  1. 비율(proportion):        이 군집 patch가 전체의 몇 %인가 (기존 히스토그램과 동일)
  2. dispersion:               이 군집 patch들이 슬라이드 내에서 공간적으로 얼마나 퍼져있는가
                               (models/spatial_features.py::attention_dispersion과 동일 공식,
                               가중치를 attention 대신 "이 군집 소속 여부(균등)"로 사용)
  3. 군집 내부 임베딩 분산:    이 환자의 이 군집 patch들이 UNI2-h 임베딩 공간에서 서로 얼마나
                               다른가(자기 자신의 평균 기준 분산) — 원래 "분화도 heterogeneity"가
                               노리던 것의 라벨-프리 대체 통계(진짜 분화도는 아님)
  4. 전역 중심까지 거리:       이 환자의 이 군집 patch들이 전체 코호트 기준 그 군집의 "전형적인"
                               모습(data/cluster_centroids_uni2native.pt)에서 얼마나 떨어져
                               있는가 — 원래 "성숙도"가 노리던 것의 라벨-프리 대체 통계(진짜
                               성숙도는 아님)

[배경, 2026-09-01] "분화도"/"성숙도"는 원래 별도로 학습된 classifier가 있어야 나오는 값인데,
그 학습에 필요한 라벨이 TCGA/CPTAC 어디에도 없다(clinical.tsv, pathology_detail.tsv 확인
완료) — 152개 생존 라벨만으로 새 MLP/CNN을 학습시키는 것도 이번 세션 내내 실패해온 패턴
(MCAT/PORPOISE/sharpening)과 같은 리스크라 배제(사용자 결정). 대신 이름을 "분화도"/"성숙도"가
아니라 "군집 내부 이질성"/"전역 대비 비정형성"으로 정직하게 부르고, 학습 파라미터 없이
결정론적으로 계산 가능한 대체 통계를 쓴다. 침윤전선(성장 패턴) CNN은 v2에서도 계속 제외 —
안전한 4개 통계로 먼저 신호가 있는지 확인한 뒤에 판단(사용자 결정).

환자가 슬라이드를 여러 장 가지면: dispersion(슬라이드 내부 좌표 기반)은 슬라이드별로 계산 후
patch 수 가중 평균, 나머지(비율/임베딩분산/중심거리)는 슬라이드 구분 없이 그 환자의 해당
군집 patch 전체를 풀링해 계산.

사용법:
    python -m data.compute_cluster_features_uni2native
    python -m data.compute_cluster_features_uni2native --datasets tcga
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
    "tcga": _ROOT / "data" / "cluster_features_uni2native_tcga.csv",
    "cptac": _ROOT / "data" / "cluster_features_uni2native_cptac.csv",
}
CASE_ID_TOKENS = {"tcga": 3, "cptac": 2}


def _case_id_from_stem(stem: str, dataset: str) -> str:
    parts = stem.split("-")
    n = CASE_ID_TOKENS[dataset]
    return "-".join(parts[:n])


def _slide_dispersion_per_cluster(assign: np.ndarray, coords: np.ndarray, k: int) -> np.ndarray:
    """models/spatial_features.py::attention_dispersion과 동일 공식(가중 좌표 표준편차,
    row/col 평균) — 가중치를 "이 군집 소속이면 균등, 아니면 0"으로 준다. 군집이 이 슬라이드에
    없으면 0."""
    out = np.zeros(k, dtype=np.float64)
    for ci in range(k):
        mask = assign == ci
        n = mask.sum()
        if n < 2:
            continue
        pts = coords[mask].astype(np.float64)
        centroid = pts.mean(axis=0)
        var = ((pts - centroid) ** 2).mean(axis=0)  # (2,) row/col 분산
        out[ci] = float(np.sqrt(np.clip(var, 0, None)).mean())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default="tcga,cptac")
    args = parser.parse_args()

    centroids = torch.load(CENTROIDS_PATH).numpy().astype(np.float64)  # (K, 1536)
    k = centroids.shape[0]
    print(f"K={k} 군집 중심 로드: {CENTROIDS_PATH}")

    for ds in args.datasets.split(","):
        ds_dir = FEATURES_ROOT / ds
        h5_paths = sorted(ds_dir.glob("*.h5"))
        print(f"\n=== {ds}: {len(h5_paths)}개 슬라이드 ===")

        # case_id -> 누적 상태
        counts: dict[str, np.ndarray] = {}                     # (K,) raw count
        disp_weighted_sum: dict[str, np.ndarray] = {}           # (K,) dispersion*n_patch_in_slide 누적
        disp_weight_total: dict[str, np.ndarray] = {}           # (K,) n_patch_in_slide 누적(dispersion 정규화용)
        feat_sum: dict[str, np.ndarray] = {}                    # (K, 1536) 군집별 patch feature 합(자기평균/중심거리용)
        feat_sqsum: dict[str, np.ndarray] = {}                  # (K,) 군집별 patch feature 제곱합(자기평균 기준 분산용, 스칼라화)
        cent_dist_sum: dict[str, np.ndarray] = {}                # (K,) 전역 중심까지 거리 누적

        for h5_path in h5_paths:
            case_id = _case_id_from_stem(h5_path.stem, ds)
            with h5py.File(h5_path, "r") as f:
                feat = f["features"][0].astype(np.float64)  # (N, 1536)
                coords = f["coords"][0]                       # (N, 2)
            d = np.linalg.norm(feat[:, None, :] - centroids[None, :, :], axis=-1)  # (N, K)
            assign = d.argmin(axis=1)
            n = feat.shape[0]
            cnt = np.bincount(assign, minlength=k).astype(np.float64)

            slide_disp = _slide_dispersion_per_cluster(assign, coords, k)

            if case_id not in counts:
                counts[case_id] = np.zeros(k)
                disp_weighted_sum[case_id] = np.zeros(k)
                disp_weight_total[case_id] = np.zeros(k)
                feat_sum[case_id] = np.zeros((k, feat.shape[1]))
                feat_sqsum[case_id] = np.zeros(k)
                cent_dist_sum[case_id] = np.zeros(k)

            counts[case_id] += cnt
            disp_weighted_sum[case_id] += slide_disp * cnt
            disp_weight_total[case_id] += cnt

            for ci in range(k):
                mask = assign == ci
                if not mask.any():
                    continue
                pts = feat[mask]  # (n_ci, 1536)
                feat_sum[case_id][ci] += pts.sum(axis=0)
                feat_sqsum[case_id][ci] += (pts ** 2).sum()  # 스칼라 누적(자기평균 분산 계산용)
                cent_dist_sum[case_id][ci] += np.linalg.norm(pts - centroids[ci], axis=1).sum()

        rows = []
        for case_id in sorted(counts.keys()):
            cnt = counts[case_id]
            total = cnt.sum()
            proportion = cnt / total if total > 0 else cnt

            disp = np.divide(disp_weighted_sum[case_id], disp_weight_total[case_id],
                              out=np.zeros(k), where=disp_weight_total[case_id] > 0)

            # 군집 내부 임베딩 분산(자기 평균 기준, 스칼라화: mean ||x - mean_ci||^2 per dim 평균) —
            # E[||x||^2] - ||E[x]||^2 항등식으로 patch를 다시 순회하지 않고 누적값만으로 계산.
            intra_var = np.zeros(k)
            cent_dist = np.zeros(k)
            for ci in range(k):
                n_ci = cnt[ci]
                if n_ci < 1:
                    continue
                mean_ci = feat_sum[case_id][ci] / n_ci
                mean_sq_norm = feat_sqsum[case_id][ci] / n_ci
                var = mean_sq_norm - float(np.dot(mean_ci, mean_ci))
                intra_var[ci] = max(var, 0.0) / feat_sum[case_id].shape[1]  # 차원 평균(스케일 안정화)
                cent_dist[ci] = cent_dist_sum[case_id][ci] / n_ci

            row = {"case_id": case_id, "n_patches": int(total)}
            for ci in range(k):
                row[f"prop_{ci}"] = proportion[ci]
                row[f"disp_{ci}"] = disp[ci]
                row[f"intravar_{ci}"] = intra_var[ci]
                row[f"centdist_{ci}"] = cent_dist[ci]
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = OUT_PATHS[ds]
        df.to_csv(out_path, index=False)
        n_feat_cols = 4 * k
        print(f"  {len(df)}명 환자 -> {out_path} ({n_feat_cols}차원 feature: prop/disp/intravar/centdist x K={k})")


if __name__ == "__main__":
    main()
