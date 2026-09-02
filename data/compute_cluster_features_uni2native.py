"""
HDP(Human Doctor Prognosis) 모델의 feature 전처리 v3 — v2(hard nearest-centroid assignment)를
soft cluster membership으로 바꿔 4가지 라벨-프리 통계를 다시 계산한다(4*K차원):

  1. 비율(proportion):        이 군집에 대한 soft weight 합이 전체의 몇 %인가
  2. dispersion:               soft weight로 가중한 좌표 표준편차(models/spatial_features.py::
                               attention_dispersion과 동일 공식 — 원래 이 함수가 attention
                               weight를 가중치로 쓰도록 설계됐던 것과 정확히 같은 형태)
  3. 군집 내부 임베딩 분산:    soft weight로 가중한 feature 분산(자기 가중평균 기준) — 원래
                               "분화도 heterogeneity"가 노리던 것의 라벨-프리 대체 통계
  4. 전역 중심까지 거리:       soft weight로 가중한, 전역 군집 중심까지의 평균 거리 — 원래
                               "성숙도"가 노리던 것의 라벨-프리 대체 통계

[배경, 2026-09-01] v2는 hard nearest-centroid(argmin)라 경계 근처 patch가 노이즈에 취약하다는
지적(사용자) — patch마다 K개 군집까지의 거리를 **per-patch adaptive temperature softmax**로
부드럽게 만든다: z[i,:] = (d[i,:]-mean_k(d[i,:]))/std_k(d[i,:]), w[i,:] = softmax(-z[i,:]).
거리 절대 스케일과 무관하게(각 patch 내에서 "상대적으로 얼마나 가까운가"만 보므로), 확실히
가까운 patch는 여전히 거의 one-hot에 가깝고 경계에 걸친 patch만 여러 군집에 걸쳐 나뉜다.
학습 파라미터는 여전히 0개 — k-means 중심(고정)에 대한 거리 계산의 연속 완화(relaxation)일 뿐.

환자가 슬라이드를 여러 장 가지면: dispersion(슬라이드 내부 좌표 기반)은 슬라이드별로 계산 후
soft weight 합으로 가중 평균, 나머지(비율/임베딩분산/중심거리)는 슬라이드 구분 없이 그 환자의
전체 patch를 soft weight로 풀링해 계산.

사용법:
    python -m data.compute_cluster_features_uni2native
    python -m data.compute_cluster_features_uni2native --datasets tcga
    python -m data.compute_cluster_features_uni2native --temperature-eps 1e-6
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


def _soft_weights(dist: np.ndarray, eps: float) -> np.ndarray:
    """dist: (N, K) patch-to-centroid 거리. patch별(row별) adaptive temperature softmax.

    Returns: (N, K) — row별 합=1인 soft membership weight.
    """
    mu = dist.mean(axis=1, keepdims=True)
    sigma = dist.std(axis=1, keepdims=True)
    z = (dist - mu) / (sigma + eps)
    z = z - z.min(axis=1, keepdims=True)  # overflow 방지(exp 인자를 <=0로)
    w = np.exp(-z)
    w = w / w.sum(axis=1, keepdims=True)
    return w


def _slide_dispersion_per_cluster(w: np.ndarray, coords: np.ndarray, k: int) -> np.ndarray:
    """models/spatial_features.py::attention_dispersion과 동일 공식(가중 좌표 표준편차,
    row/col 평균) — w[:, ci]를 그대로 attn_weights처럼 쓴다. 이 슬라이드에서 군집 ci에 대한
    soft weight 총합이 0에 가까우면(사실상 없음) 0."""
    out = np.zeros(k, dtype=np.float64)
    coords_f = coords.astype(np.float64)
    for ci in range(k):
        wc = w[:, ci]
        total = wc.sum()
        if total < 1e-6:
            continue
        ww = wc / total
        centroid = (ww[:, None] * coords_f).sum(axis=0)
        diff = coords_f - centroid
        var = (ww[:, None] * diff ** 2).sum(axis=0)  # (2,) row/col 분산
        out[ci] = float(np.sqrt(np.clip(var, 0, None)).mean())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default="tcga,cptac")
    parser.add_argument("--temperature-eps", type=float, default=1e-6,
                         help="per-patch std가 0에 가까울 때(K개 중심까지 거리가 전부 똑같은 "
                              "극단적 경우) 0-division 방지용 epsilon.")
    args = parser.parse_args()

    centroids = torch.load(CENTROIDS_PATH).numpy().astype(np.float64)  # (K, 1536)
    k = centroids.shape[0]
    print(f"K={k} 군집 중심 로드: {CENTROIDS_PATH}")

    for ds in args.datasets.split(","):
        ds_dir = FEATURES_ROOT / ds
        h5_paths = sorted(ds_dir.glob("*.h5"))
        print(f"\n=== {ds}: {len(h5_paths)}개 슬라이드 ===")

        weight_sum: dict[str, np.ndarray] = {}          # (K,) soft weight 총합(=soft 환자 patch 수)
        disp_weighted_sum: dict[str, np.ndarray] = {}    # (K,) dispersion*슬라이드soft weight합 누적
        disp_weight_total: dict[str, np.ndarray] = {}    # (K,) 슬라이드별 soft weight합 누적(dispersion 정규화용)
        feat_wsum: dict[str, np.ndarray] = {}             # (K, 1536) soft weight * feature 누적(가중평균용)
        feat_wsqsum: dict[str, np.ndarray] = {}            # (K,) soft weight * ||feature||^2 누적(가중분산용)
        cent_dist_wsum: dict[str, np.ndarray] = {}          # (K,) soft weight * 전역중심거리 누적

        for h5_path in h5_paths:
            case_id = _case_id_from_stem(h5_path.stem, ds)
            with h5py.File(h5_path, "r") as f:
                feat = f["features"][0].astype(np.float64)  # (N, 1536)
                coords = f["coords"][0]                       # (N, 2)
            dist = np.linalg.norm(feat[:, None, :] - centroids[None, :, :], axis=-1)  # (N, K)
            w = _soft_weights(dist, args.temperature_eps)  # (N, K), row-sum=1

            slide_disp = _slide_dispersion_per_cluster(w, coords, k)
            slide_weight_total = w.sum(axis=0)  # (K,)

            if case_id not in weight_sum:
                weight_sum[case_id] = np.zeros(k)
                disp_weighted_sum[case_id] = np.zeros(k)
                disp_weight_total[case_id] = np.zeros(k)
                feat_wsum[case_id] = np.zeros((k, feat.shape[1]))
                feat_wsqsum[case_id] = np.zeros(k)
                cent_dist_wsum[case_id] = np.zeros(k)

            weight_sum[case_id] += slide_weight_total
            disp_weighted_sum[case_id] += slide_disp * slide_weight_total
            disp_weight_total[case_id] += slide_weight_total

            # (K, 1536) = w^T @ feat ; (K,) = sum_i w[i,ci] * ||feat[i]||^2 ; (K,) = sum_i w[i,ci]*dist[i,ci]
            feat_wsum[case_id] += w.T @ feat
            sqnorm = (feat ** 2).sum(axis=1)  # (N,)
            feat_wsqsum[case_id] += (w * sqnorm[:, None]).sum(axis=0)
            cent_dist_wsum[case_id] += (w * dist).sum(axis=0)

        rows = []
        for case_id in sorted(weight_sum.keys()):
            wsum = weight_sum[case_id]
            total = wsum.sum()
            proportion = wsum / total if total > 0 else wsum

            disp = np.divide(disp_weighted_sum[case_id], disp_weight_total[case_id],
                              out=np.zeros(k), where=disp_weight_total[case_id] > 1e-6)

            intra_var = np.zeros(k)
            cent_dist = np.zeros(k)
            for ci in range(k):
                wt = wsum[ci]
                if wt < 1e-6:
                    continue
                mean_ci = feat_wsum[case_id][ci] / wt
                mean_sq_norm = feat_wsqsum[case_id][ci] / wt
                var = mean_sq_norm - float(np.dot(mean_ci, mean_ci))
                intra_var[ci] = max(var, 0.0) / feat_wsum[case_id].shape[1]
                cent_dist[ci] = cent_dist_wsum[case_id][ci] / wt

            row = {"case_id": case_id, "n_patches_soft": float(total)}
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
        print(f"  {len(df)}명 환자 -> {out_path} ({n_feat_cols}차원 feature: prop/disp/intravar/centdist x K={k}, soft weight)")


if __name__ == "__main__":
    main()
