"""
M1(WSI 단독, coord-embed-concat 적용, internal 0.5352/external 0.5254)과 M5_STG_R(Clinical
age/sex+margin+staging 단독)을 완전히 독립적으로 학습된 채로 두고, 최종 risk score만 Cox
선형결합(risk = α·risk_M1 + β·risk_M5)한다 — scripts/stack_m1pool_m5_cox.py와 동일 방법론.

배경(2026-08-15): M2(WSI+clinical을 cox_add나 concat으로 "같이" 학습)에 coord-embed-concat을
얹으니 cox_add에서 clinical_linear가 gradient는 건강한데 순 기여가 0으로 수렴하는 현상이
fold0/fold1 양쪽에서 재현 확인됐다("같이 학습하면 clinical이 밀려난다"는 stack_m1pool_m5_cox.py
때의 관찰과 같은 클래스). 그렇다면 아예 두 모델을 따로 학습시킨 뒤 risk score 레벨에서만
합치면 서로 간섭 없이 각자 최선을 낼 수 있는지 확인한다.

M1/M5 둘 다 이미 seed42 5-fold로 완전히 학습되어 kfold_preds/external_preds CSV가 저장돼
있어(재학습 없이) 순수 결합만 한다:
  - M1:     .logs/kfold_preds/tcga_M1_uni2_DISP_NOVIT_COORD_CAT_FOLD*OF5_seed42_*.csv
  - M5_STG_R: .logs/kfold_preds/tcga_M5_STG_R_FOLD*OF5_seed42_*.csv

[Internal] 두 모델의 5-fold OOF risk를 case_id로 병합해 CoxPHFitter로 계수를 적합한 뒤 같은
데이터에 대해 결합 risk를 계산한다(계수를 적합한 데이터로 바로 평가하는 낙관적 편향 있음 —
파라미터 2개뿐이라 크지 않을 것으로 보이나 명확히 표시한다).

[External] 두 모델 각각 5-fold external_preds(폴드별 체크포인트로 CPTAC 전체 평가한 것)를
평균(ensemble mean)한 뒤 위에서 적합한 (α, β)로 결합한다.

사용법: python -m scripts.stack_m1coordcat_m5_cox
"""
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.metrics import compute_survival_metrics

try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

N_FOLDS = 5
SEED = 42
KFOLD_PREDS_DIR = _ROOT / ".logs" / "kfold_preds"
EXTERNAL_PREDS_DIR = _ROOT / ".logs" / "external_preds"

M1_TAG = "M1_uni2_DISP_NOVIT_COORD_CAT"
M5_TAG = "M5_STG_R"


def load_internal_oof(model_tag: str) -> pd.DataFrame:
    dfs = []
    for fold in range(N_FOLDS):
        path = KFOLD_PREDS_DIR / f"tcga_{model_tag}_FOLD{fold}OF{N_FOLDS}_seed{SEED}_fold{fold}of{N_FOLDS}.csv"
        dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)


def load_external_ensemble(model_tag: str) -> pd.DataFrame:
    """5개 fold 체크포인트로 각각 평가된 external risk를 case_id별로 평균(ensemble mean)한다."""
    dfs = []
    for fold in range(N_FOLDS):
        path = EXTERNAL_PREDS_DIR / f"cptac_{model_tag}_FOLD{fold}OF{N_FOLDS}_seed{SEED}_fold{fold}of{N_FOLDS}.csv"
        df = pd.read_csv(path)[["case_id", "risk", "OS_time", "OS_event"]]
        dfs.append(df.rename(columns={"risk": f"risk_fold{fold}"}))
    merged = dfs[0][["case_id", "OS_time", "OS_event"]].copy()
    risk_cols = []
    for fold, df in enumerate(dfs):
        merged = merged.merge(df[["case_id", f"risk_fold{fold}"]], on="case_id", how="inner")
        risk_cols.append(f"risk_fold{fold}")
    merged["risk"] = merged[risk_cols].mean(axis=1)
    return merged[["case_id", "risk", "OS_time", "OS_event"]]


def main():
    if not LIFELINES_AVAILABLE:
        raise RuntimeError("lifelines가 필요합니다.")

    print("=== Internal OOF risk 로드 ===")
    m1_internal = load_internal_oof(M1_TAG)[["case_id", "risk", "OS_time", "OS_event"]].rename(columns={"risk": "risk_m1"})
    m5_internal = load_internal_oof(M5_TAG)[["case_id", "risk"]].rename(columns={"risk": "risk_m5"})
    merged = m1_internal.merge(m5_internal, on="case_id", how="inner")
    print(f"내부 병합 환자 수: {len(merged)} (M1={len(m1_internal)}, M5={len(m5_internal)})")

    baseline_m1 = compute_survival_metrics(merged["risk_m1"].values, merged["OS_time"].values, merged["OS_event"].values)
    baseline_m5 = compute_survival_metrics(merged["risk_m5"].values, merged["OS_time"].values, merged["OS_event"].values)
    print(f"  (참고) M1 단독 internal: C={baseline_m1['c_index']:.4f}")
    print(f"  (참고) M5_STG_R 단독 internal: C={baseline_m5['c_index']:.4f}")

    print("\n=== Cox 결합 계수(α, β) 적합 (internal OOF risk 전체 사용) ===")
    cph = CoxPHFitter()
    cph.fit(
        pd.DataFrame({"time": merged["OS_time"], "event": merged["OS_event"],
                      "risk_m1": merged["risk_m1"], "risk_m5": merged["risk_m5"]}),
        duration_col="time", event_col="event",
    )
    alpha = float(cph.params_["risk_m1"])
    beta = float(cph.params_["risk_m5"])
    print(f"  alpha(risk_m1 계수)={alpha:.4f}  beta(risk_m5 계수)={beta:.4f}")

    combined_internal_risk = alpha * merged["risk_m1"].values + beta * merged["risk_m5"].values
    combined_internal_metrics = compute_survival_metrics(combined_internal_risk, merged["OS_time"].values, merged["OS_event"].values)
    print(f"\n  결합 internal(같은 데이터로 계수 적합 + 평가라 낙관적 편향 있음 주의): "
          f"C={combined_internal_metrics['c_index']:.4f}  HR={combined_internal_metrics['hr']:.3f}  "
          f"logrank_p={combined_internal_metrics['log_rank_p']:.4f}")

    print("\n=== External risk 로드 (5-fold ensemble mean, 이미 저장된 CSV 사용) ===")
    m1_ext = load_external_ensemble(M1_TAG).rename(columns={"risk": "risk_m1"})
    m5_ext = load_external_ensemble(M5_TAG)[["case_id", "risk"]].rename(columns={"risk": "risk_m5"})
    ext_df = m1_ext.merge(m5_ext, on="case_id", how="inner")
    print(f"external 병합 환자 수: {len(ext_df)}")

    baseline_m1_ext = compute_survival_metrics(ext_df["risk_m1"].values, ext_df["OS_time"].values, ext_df["OS_event"].values)
    baseline_m5_ext = compute_survival_metrics(ext_df["risk_m5"].values, ext_df["OS_time"].values, ext_df["OS_event"].values)
    print(f"  (참고) M1 단독 external(ensemble mean): C={baseline_m1_ext['c_index']:.4f}")
    print(f"  (참고) M5_STG_R 단독 external(ensemble mean): C={baseline_m5_ext['c_index']:.4f}")

    combined_ext_risk = alpha * ext_df["risk_m1"].values + beta * ext_df["risk_m5"].values
    combined_ext_metrics = compute_survival_metrics(combined_ext_risk, ext_df["OS_time"].values, ext_df["OS_event"].values)
    print(f"\n  결합 external: C={combined_ext_metrics['c_index']:.4f}  HR={combined_ext_metrics['hr']:.3f}  "
          f"logrank_p={combined_ext_metrics['log_rank_p']:.4f}")


if __name__ == "__main__":
    main()
