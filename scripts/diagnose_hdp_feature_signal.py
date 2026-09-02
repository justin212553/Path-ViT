"""
HDP WSI feature(들)가 신경망 없이 그 자체로 생존과 상관이 있는지 진단한다.

배경(2026-09-01): HDP_Pretrain_Cluster(uni2native, prop/content_entropy 포함) 2seed x 5fold
검증 결과 internal~0.63/external~0.61로 M7(RNA+Clinical only)과 통계적으로 동일 — 사용자가
"모델 내에서 무슨 일이 일어나는지 알고 싶다"고 요청. 가장 먼저 분리해야 할 질문: 조인트
모델이 학습을 못한 건지, 아니면 UNI2-h가 뽑은 feature 자체에 애초에 생존 신호가 없는 건지.

이 스크립트는 신경망을 전혀 안 거치고, scripts/apply_hdp_pretrain_head.py /
data/compute_cluster_features_uni2native.py가 만든 raw feature 컬럼 하나하나를 그대로
"risk score"로 취급해 utils/metrics.py::compute_survival_metrics로 c-index/HR/log-rank p를
계산한다 — 방향이 반대일 수 있어 원값과 부호 반전 둘 다 본다(max가 실제 판별력).
0.5 근처(둘 다)면 그 feature 단독으로는 생존과 무관하다는 뜻.

사용법:
    python -m scripts.diagnose_hdp_feature_signal --dataset tcga
    python -m scripts.diagnose_hdp_feature_signal --dataset cptac
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.metrics import compute_survival_metrics

PRETRAIN_PATHS = {
    "tcga": Path("data/tumor_content_uni2native_tcga.csv"),
    "cptac": Path("data/tumor_content_uni2native_cptac.csv"),
}
CLUSTER_PATHS = {
    "tcga": Path("data/cluster_features_uni2native_tcga.csv"),
    "cptac": Path("data/cluster_features_uni2native_cptac.csv"),
}
CLINICAL_PATHS = {
    "tcga": Path("data/clinical_tcga.csv"),
    "cptac": Path("data/clinical_cptac.csv"),
}
PRETRAIN_COLS = ["mean_tumor_content", "tumor_heterogeneity", "tumor_dispersion", "frac_high_tumor", "content_entropy"]


def _report(name: str, feature: np.ndarray, times: np.ndarray, events: np.ndarray):
    m_pos = compute_survival_metrics(feature, times, events)
    m_neg = compute_survival_metrics(-feature, times, events)
    c_pos, c_neg = m_pos["c_index"], m_neg["c_index"]
    best = m_pos if c_pos >= c_neg else m_neg
    direction = "+" if c_pos >= c_neg else "-"
    best_c = max(c_pos, c_neg) if not (np.isnan(c_pos) or np.isnan(c_neg)) else float("nan")
    print(f"  {name:22s} c_index={best_c:.4f} (raw={c_pos:.4f}/inv={c_neg:.4f}, 방향={direction}) "
          f"HR={best['hr']:.3f} log_rank_p={best['log_rank_p']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    args = parser.parse_args()
    ds = args.dataset

    clinical = pd.read_csv(CLINICAL_PATHS[ds])[["case_id", "OS_time", "OS_event"]].dropna()

    print(f"=== {ds}: pretrain(PanNuke 지도학습 종양함량) feature 단독 판별력 (신경망 없음, N={len(clinical)}) ===")
    pretrain = pd.read_csv(PRETRAIN_PATHS[ds])
    merged = clinical.merge(pretrain, on="case_id", how="inner")
    print(f"  매칭된 환자 수: {len(merged)}/{len(clinical)}")
    times = merged["OS_time"].to_numpy(dtype=float)
    events = merged["OS_event"].to_numpy(dtype=int)
    for col in PRETRAIN_COLS:
        _report(col, merged[col].to_numpy(dtype=float), times, events)

    print(f"\n=== {ds}: cluster(K=10, 비지도) feature 단독 판별력 ===")
    cluster = pd.read_csv(CLUSTER_PATHS[ds])
    merged_c = clinical.merge(cluster, on="case_id", how="inner")
    print(f"  매칭된 환자 수: {len(merged_c)}/{len(clinical)}")
    times_c = merged_c["OS_time"].to_numpy(dtype=float)
    events_c = merged_c["OS_event"].to_numpy(dtype=int)
    _report("prop_entropy", merged_c["prop_entropy"].to_numpy(dtype=float), times_c, events_c)
    k = sum(1 for c in cluster.columns if c.startswith("prop_") and c != "prop_entropy")
    best_per_family = {}
    for prefix in ["prop_", "disp_", "intravar_", "centdist_"]:
        best_c, best_col = 0.5, None
        for ci in range(k):
            col = f"{prefix}{ci}"
            if col not in merged_c.columns:
                continue
            feat = merged_c[col].to_numpy(dtype=float)
            m_pos = compute_survival_metrics(feat, times_c, events_c)
            m_neg = compute_survival_metrics(-feat, times_c, events_c)
            c = max(m_pos["c_index"], m_neg["c_index"])
            if not np.isnan(c) and abs(c - 0.5) > abs(best_c - 0.5):
                best_c, best_col = c, col
        best_per_family[prefix] = (best_col, best_c)
    print(f"  K={k}개 군집 중 |c-0.5|가 가장 큰 컬럼(patch 수가 적은 군집일수록 우연히 튈 수 있음 — 참고용):")
    for prefix, (col, c) in best_per_family.items():
        print(f"    {prefix:10s} 최댓값: {col} c_index={c:.4f}")

    print(f"\n=== {ds}: pretrain 6개 feature를 선형결합(다변량 Cox, in-sample!) 했을 때 ===")
    print("  (held-out 검증 아님 — '개별로는 0.5 근처인 feature들을 자유롭게 조합해도 신호가 "
          "나오는지'만 보는 낙관적 상한선 체크)")
    combo_cols = PRETRAIN_COLS + ["prop_entropy"]
    df = merged.merge(merged_c[["case_id", "prop_entropy"]], on="case_id")[combo_cols + ["OS_time", "OS_event"]].dropna()
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df, duration_col="OS_time", event_col="OS_event")
    risk = cph.predict_partial_hazard(df).to_numpy().ravel()
    m = compute_survival_metrics(risk, df["OS_time"].to_numpy(), df["OS_event"].to_numpy().astype(int))
    print(f"  N={len(df)} in-sample c_index={m['c_index']:.4f} log_rank_p={m['log_rank_p']:.4f}")
    sig_terms = [c for c, p in zip(cph.summary.index, cph.summary["p"]) if p < 0.05]
    print(f"  개별 계수 중 p<0.05: {sig_terms if sig_terms else '없음'}")


if __name__ == "__main__":
    main()
