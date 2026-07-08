"""
Screening 성능 평가 지표
"""
import numpy as np
from sklearn.metrics import roc_curve, auc


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
    생존 분석 성능 지표: Harrell's concordance index (c-index).

    모든 comparable 환자 쌍(더 이른 시점에 사망(event=1)한 환자 vs 그 시점 이후까지
    관찰된 다른 환자)에 대해, risk score의 대소 관계가 실제 생존 순서와 일치하는 비율.
    동점(risk score가 같은 쌍)은 0.5로 부분 반영한다.

    Args:
        risk_scores: (N,) 예측 risk score (클수록 위험/사망 가능성 높음)
        times:       (N,) OS_time
        events:      (N,) OS_event (1=사망, 0=censored)

    Returns:
        dict: c_index (comparable pair가 하나도 없으면 nan)
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
    return {"c_index": c_index}
