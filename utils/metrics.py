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
    Youden's J statistic(TPR - FPR 최대화)으로 최적 threshold를 먼저 구한 뒤
    해당 threshold로 이진 예측을 만들어 F1/precision/recall/accuracy를 계산한다.

    Args:
        patch_scores: (N,) softmax 양성 확률 [0, 1]
        patch_labels: (N,) GT 라벨 (0 또는 1)

    Returns:
        dict: accuracy, auc_roc, f1, precision, recall, threshold (Youden's J)
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    # 1) Youden's J로 최적 threshold 계산
    try:
        fpr, tpr, thresholds = roc_curve(patch_labels, patch_scores)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_threshold = float(thresholds[best_idx])
        auc_roc = float(auc(fpr, tpr))
    except ValueError:
        best_threshold = 0.5
        auc_roc = float("nan")

    # 2) 최적 threshold로 이진 예측 생성
    patch_preds = (patch_scores >= best_threshold).astype(np.int64)

    # 3) 분류 지표 계산
    return {
        "threshold": best_threshold,
        "accuracy":  float(accuracy_score(patch_labels, patch_preds)),
        "f1":        float(f1_score(patch_labels, patch_preds, zero_division=0)),
        "precision": float(precision_score(patch_labels, patch_preds, zero_division=0)),
        "recall":    float(recall_score(patch_labels, patch_preds, zero_division=0)),
        "auc_roc":   auc_roc,
    }
