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
