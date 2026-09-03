"""
HDP_Pretrain 2단계 — scripts/train_hdp_pretrain_head.py로 PanNuke에서 학습시킨 종양 함량
회귀 head를, 우리 코호트(TCGA-PAAD/CPTAC-PDA)의 이미 추출된 uni2native feature(data/
uni2h_official_features/*.h5)에 그대로 적용한다. 원본 WSI 이미지를 다시 열 필요 없음 —
h5에 이미 있는 1536차원 feature에 이 head(frozen)를 forward만 하면 된다.

환자별로 5가지 스칼라를 계산한다(cluster 버전의 4*K차원과 대응되지만, 이번엔 "클래스"가
비지도 군집 10개가 아니라 진짜 의미가 있는 단일 축(종양 함량)이라 훨씬 저차원으로 끝난다):
  1. mean_tumor_content:  환자의 전체 patch 평균 종양 함량 — 원래 "종양 비율"이 노리던 것
  2. tumor_heterogeneity: patch 간 종양 함량의 분산 — 원래 "분화도 heterogeneity"가 노리던 것
                          (이번엔 진짜 종양-관련 축의 분산이라 §2-1-2-1이 원래 의도했던 것에
                          훨씬 가깝다 — 군집 버전의 "군집 내부 임베딩 분산"보다 의미가 명확함)
  3. tumor_dispersion:    종양 함량으로 가중한 좌표 표준편차(models/spatial_features.py::
                          attention_dispersion과 동일 공식) — 종양이 국소적으로 뭉쳐있는지
                          슬라이드 전역에 퍼져있는지
  4. frac_high_tumor:     종양 함량 > threshold(기본 0.1)인 patch 비율 — 사람이 보기 쉬운
                          보조 지표(0에 가까우면 "이 환자는 촬영된 영역에 종양이 거의 없다"는
                          뜻일 수 있음, PanNuke pancreas subset 자체가 195장 중 133장 종양
                          면적 0이었던 것과 같은 현상이 우리 코호트에도 있을 수 있음 — 정상
                          관찰).
  5. content_entropy:     종양 함량 값을 N_BINS(기본 10)개 구간으로 나눠, patch들이 그
                          구간들에 얼마나 고르게/쏠려서 분포하는지의 정규화 Shannon entropy
                          (0=한 구간에 다 몰림, 1=완전 균등분포). 2026-09-01, Takamatsu et al.
                          2026(Br J Cancer, PAAD 591+302명, H&E CNN 클러스터 17개의 비율
                          분포 엔트로피가 재발 시점과 독립적으로 유의했다는 보고, P<0.01)의
                          "클러스터 구성 이질성" 통계를 참고 — 우리는 클러스터가 아니라 연속
                          종양 함량 스칼라라 binning으로 근사한다. mean_tumor_content/
                          tumor_heterogeneity(분산)와 달리, "종양이 있다/없다"가 아니라
                          "종양 함량 자체가 patch마다 얼마나 들쭉날쭉한가"를 재는 별도 축.

환자가 슬라이드를 여러 장 가지면 슬라이드 구분 없이 그 환자의 전체 patch를 풀링해 계산(단
dispersion만 슬라이드 내부 좌표계가 필요해 슬라이드별로 계산 후 종양함량 가중 평균).

사용법:
    python -m scripts.apply_hdp_pretrain_head
    python -m scripts.apply_hdp_pretrain_head --high-threshold 0.15
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

from models.tumor_content_head import TumorContentHead

FEATURES_ROOT = _ROOT / "data" / "uni2h_official_features"
# 2026-09-02: FEATURES_ROOT(h5)는 MahmoodLab 공식 배포가 DX(진단용) 슬라이드만 담고 있어
# TCGA는 슬라이드 커버리지가 203/466(44%)뿐이었다 — TS/BS 슬라이드가 통째로 빠짐(환자 2명은
# DX 자체가 없어 이전엔 그 환자 전체가 fallback 처리됐음). HPC에서 tile 트리 전체(featur es_
# uni2native.pt/coords_uni2native.pt, 모든 슬라이드 타입)를 로컬로 옮겨온 뒤(scripts/
# zip_uni2native_for_local.sh) --source tiletree로 전체 슬라이드를 쓸 수 있게 함.
TILES_ROOT = {"tcga": _ROOT / "data" / "patches_tcga" / "tiles", "cptac": _ROOT / "data" / "patches_cptac" / "tiles"}
# models/checkpoint/는 git-ignore 대상이라 data/ 밑에 별도로 둔다(2026-09-01, HPC에서
# train_hdp_pretrain_cluster.py가 이 파일을 못 찾는 문제로 확인 — git pull만으로 따라오게).
HEAD_PATH_DEFAULT = _ROOT / "data" / "hdp_pretrain_tumor_content_head.pt"
CASE_ID_TOKENS = {"tcga": 3, "cptac": 2}
N_BINS = 10  # content_entropy용 종양 함량(0~1) 구간 수 — 원래 군집 K=10과 맞춤(비교 편의)


def _out_paths(tag: str) -> dict:
    return {
        "tcga": _ROOT / "data" / f"tumor_content_uni2native{tag}_tcga.csv",
        "cptac": _ROOT / "data" / f"tumor_content_uni2native{tag}_cptac.csv",
    }


def _case_id_from_stem(stem: str, dataset: str) -> str:
    parts = stem.split("-")
    n = CASE_ID_TOKENS[dataset]
    return "-".join(parts[:n])


def _load_head(device, head_path: Path) -> TumorContentHead:
    ckpt = torch.load(head_path, map_location=device, weights_only=False)
    head = TumorContentHead(in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"]).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    head.requires_grad_(False)
    print(f"head 로드: {head_path} (best_val_loss={ckpt['best_val_loss']:.4f}, "
          f"n_train={ckpt['n_train']}, n_val={ckpt['n_val']})")
    return head


def _iter_slides(ds: str, source: str):
    """(case_id, feat(N,1536) np.ndarray, coords(N,2) np.ndarray)를 슬라이드 단위로 yield.
    source="h5": 기존 MahmoodLab 공식 배포(DX만). source="tiletree": 로컬/HPC에 있는 uni2native
    tile 트리 전체(모든 슬라이드 타입) — scripts/zip_uni2native_for_local.sh로 받은 것."""
    if source == "h5":
        import h5py
        h5_paths = sorted((FEATURES_ROOT / ds).glob("*.h5"))
        print(f"\n=== {ds}: {len(h5_paths)}개 슬라이드(h5, DX만) ===")
        for h5_path in h5_paths:
            case_id = _case_id_from_stem(h5_path.stem, ds)
            with h5py.File(h5_path, "r") as f:
                yield case_id, f["features"][0], f["coords"][0]
    else:
        slide_dirs = sorted(d for d in TILES_ROOT[ds].iterdir()
                             if d.is_dir() and (d / "features_uni2native.pt").exists())
        print(f"\n=== {ds}: {len(slide_dirs)}개 슬라이드(tile 트리, 전체 타입) ===")
        for slide_dir in slide_dirs:
            case_id = _case_id_from_stem(slide_dir.name, ds)
            feat = torch.load(slide_dir / "features_uni2native.pt", weights_only=True).numpy()
            coords = torch.load(slide_dir / "coords_uni2native.pt", weights_only=True).numpy()
            yield case_id, feat, coords


def _slide_dispersion(coords: np.ndarray, weights: np.ndarray) -> float:
    """models/spatial_features.py::attention_dispersion과 동일 공식 — weights(종양 함량)로
    가중한 좌표 표준편차."""
    total = weights.sum()
    if total < 1e-6:
        return 0.0
    w = weights / total
    coords_f = coords.astype(np.float64)
    centroid = (w[:, None] * coords_f).sum(axis=0)
    diff = coords_f - centroid
    var = (w[:, None] * diff ** 2).sum(axis=0)
    return float(np.sqrt(np.clip(var, 0, None)).mean())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default="tcga,cptac")
    parser.add_argument("--high-threshold", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--head-path", type=str, default=str(HEAD_PATH_DEFAULT),
                         help="2026-09-02: 다른 head(예: 해상도 보정판 hdp_pretrain_tumor_content_"
                              "head_resmatch.pt)로 재생성할 때 지정. --out-tag와 함께 써야 기존 "
                              "산출물을 안 덮어씀.")
    parser.add_argument("--out-tag", type=str, default="",
                         help="출력 파일명에 붙일 접미사(예: '_resmatch' -> "
                              "tumor_content_uni2native_resmatch_tcga.csv). 기본 ''=기존 파일명 그대로.")
    parser.add_argument("--source", type=str, default="h5", choices=["h5", "tiletree"],
                         help="h5(기본, 기존 동작, DX만) | tiletree(2026-09-02 신규 — 전체 슬라이드 "
                              "타입, zip_uni2native_for_local.sh로 로컬에 받아온 경우에만 가능).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = _load_head(device, Path(args.head_path))
    out_paths = _out_paths(args.out_tag)

    for ds in args.datasets.split(","):
        patch_sum: dict[str, float] = {}
        patch_sqsum: dict[str, float] = {}
        patch_count: dict[str, int] = {}
        high_count: dict[str, int] = {}
        disp_weighted_sum: dict[str, float] = {}
        disp_weight_total: dict[str, float] = {}
        bin_counts: dict[str, np.ndarray] = {}

        for case_id, feat, coords in _iter_slides(ds, args.source):
            n = feat.shape[0]
            scores = np.empty(n, dtype=np.float64)
            for i in range(0, n, args.batch_size):
                chunk = torch.from_numpy(feat[i:i + args.batch_size]).float().to(device)
                scores[i:i + args.batch_size] = head(chunk).cpu().numpy()

            if case_id not in patch_sum:
                patch_sum[case_id] = 0.0
                patch_sqsum[case_id] = 0.0
                patch_count[case_id] = 0
                high_count[case_id] = 0
                disp_weighted_sum[case_id] = 0.0
                disp_weight_total[case_id] = 0.0
                bin_counts[case_id] = np.zeros(N_BINS, dtype=np.int64)

            patch_sum[case_id] += float(scores.sum())
            patch_sqsum[case_id] += float((scores ** 2).sum())
            patch_count[case_id] += n
            high_count[case_id] += int((scores > args.high_threshold).sum())

            slide_disp = _slide_dispersion(coords, scores)
            slide_weight = float(scores.sum())
            disp_weighted_sum[case_id] += slide_disp * slide_weight
            disp_weight_total[case_id] += slide_weight

            bin_idx = np.clip((scores * N_BINS).astype(np.int64), 0, N_BINS - 1)
            bin_counts[case_id] += np.bincount(bin_idx, minlength=N_BINS)

        rows = []
        for case_id in sorted(patch_sum.keys()):
            n = patch_count[case_id]
            mean_tc = patch_sum[case_id] / n
            var_tc = max(patch_sqsum[case_id] / n - mean_tc ** 2, 0.0)
            frac_high = high_count[case_id] / n
            disp = (disp_weighted_sum[case_id] / disp_weight_total[case_id]
                    if disp_weight_total[case_id] > 1e-6 else 0.0)

            p = bin_counts[case_id] / bin_counts[case_id].sum()
            raw_entropy = -(p[p > 0] * np.log(p[p > 0])).sum()
            content_entropy = float(raw_entropy / np.log(N_BINS))  # 0~1 정규화

            rows.append({
                "case_id": case_id, "n_patches": n,
                "mean_tumor_content": mean_tc,
                "tumor_heterogeneity": var_tc,
                "tumor_dispersion": disp,
                "frac_high_tumor": frac_high,
                "content_entropy": content_entropy,
            })

        df = pd.DataFrame(rows)
        out_path = out_paths[ds]
        df.to_csv(out_path, index=False)
        print(f"  {len(df)}명 환자 -> {out_path}")
        print(f"  mean_tumor_content 분포: mean={df['mean_tumor_content'].mean():.4f} "
              f"std={df['mean_tumor_content'].std():.4f} "
              f"min={df['mean_tumor_content'].min():.4f} max={df['mean_tumor_content'].max():.4f}")


if __name__ == "__main__":
    main()
