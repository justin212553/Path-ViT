"""
Screening 성능 평가 지표
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False


def compute_patch_metrics(
    patch_scores: np.ndarray,
    patch_labels: np.ndarray,
) -> dict:
    """
    패치 레벨 분류 성능 지표.
    threshold는 0.5로 고정하고, 그 외 AUC는 threshold와 무관하게 ROC 전체에서 계산한다.

    Args:
        patch_scores: (N,) softmax 양성 확률 [0, 1]
        patch_labels: (N,) GT 라벨 (0 또는 1)

    Returns:
        dict: accuracy, auc_roc, f1, precision, recall, threshold (고정값 0.5)
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    # 1) AUC: threshold와 무관하게 ROC 전체에서 계산
    try:
        fpr, tpr, _ = roc_curve(patch_labels, patch_scores)
        auc_roc = float(auc(fpr, tpr))
    except ValueError:
        auc_roc = float("nan")

    # 2) 고정 threshold로 이진 예측 생성
    threshold = 0.5
    patch_preds = (patch_scores >= threshold).astype(np.int64)

    # 3) 분류 지표 계산
    return {
        "threshold": threshold,
        "accuracy":  float(accuracy_score(patch_labels, patch_preds)),
        "f1":        float(f1_score(patch_labels, patch_preds, zero_division=0)),
        "precision": float(precision_score(patch_labels, patch_preds, zero_division=0)),
        "recall":    float(recall_score(patch_labels, patch_preds, zero_division=0)),
        "auc_roc":   auc_roc,
    }


def compute_survival_metrics(
    risk_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
) -> dict:
    """
    생존 분석 성능 지표: c-index, hazard ratio(HR), log-rank p-value.

    c_index: Harrell's concordance index. 모든 comparable 환자 쌍(더 이른 시점에
        사망(event=1)한 환자 vs 그 시점 이후까지 관찰된 다른 환자)에 대해, risk score의
        대소 관계가 실제 생존 순서와 일치하는 비율. 동점(risk score가 같은 쌍)은 0.5로
        부분 반영한다.
    hr / log_rank_p: risk score의 중앙값으로 저위험/고위험 두 그룹으로 이분화한 뒤
        - hr: 그룹(고위험=1)을 단일 공변량으로 한 Cox 모델의 hazard ratio
        - log_rank_p: 두 그룹 KM 곡선 차이에 대한 log-rank test p-value
      (병리 생존예측 보고서에서 흔히 쓰는 저위험/고위험 이분화 방식과 동일)

    Args:
        risk_scores: (N,) 예측 risk score (클수록 위험/사망 가능성 높음)
        times:       (N,) OS_time
        events:      (N,) OS_event (1=사망, 0=censored)

    Returns:
        dict: c_index, hr, log_rank_p — 계산 불가한 경우(comparable pair 없음, 표본 부족,
              lifelines 미설치 등) 해당 값은 nan
    """
    risk  = np.asarray(risk_scores, dtype=np.float64)
    time  = np.asarray(times, dtype=np.float64)
    event = np.asarray(events, dtype=bool)

    comparable  = (time[:, None] < time[None, :]) & event[:, None]
    concordant  = comparable & (risk[:, None] > risk[None, :])
    tied_risk   = comparable & (risk[:, None] == risk[None, :])

    n_permissible = int(comparable.sum())
    c_index = (
        float((concordant.sum() + 0.5 * tied_risk.sum()) / n_permissible)
        if n_permissible > 0 else float("nan")
    )

    hr, log_rank_p = float("nan"), float("nan")
    if LIFELINES_AVAILABLE and len(risk) >= 4 and event.sum() >= 2:
        high_risk = risk > np.median(risk)
        if high_risk.any() and (~high_risk).any():
            try:
                cph = CoxPHFitter()
                cph.fit(
                    pd.DataFrame({
                        "time":      time,
                        "event":     event.astype(int),
                        "high_risk": high_risk.astype(int),
                    }),
                    duration_col="time", event_col="event",
                )
                hr = float(np.exp(cph.params_["high_risk"]))

                lr = logrank_test(
                    time[high_risk], time[~high_risk],
                    event_observed_A=event[high_risk], event_observed_B=event[~high_risk],
                )
                log_rank_p = float(lr.p_value)
            except Exception:
                pass  # 표본이 작으면 Cox fit이 수렴하지 않을 수 있음 — nan으로 둔다

    return {"c_index": c_index, "hr": hr, "log_rank_p": log_rank_p}
