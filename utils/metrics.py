"""
Screening 성능 평가 지표
- Top-K Retrieval Rate: 상위 K 슬라이드 중 실제 전이 슬라이드 포함 비율
- Sensitivity at Low FPR: 위양성률 억제 상태에서의 민감도
"""
import numpy as np
from sklearn.metrics import roc_curve, auc


def compute_top_k_retrieval(
    slide_scores: np.ndarray,
    slide_labels: np.ndarray,
    k: int = 20,
) -> float:
    """
    Top-K Retrieval Rate 계산.
    모델이 전이 위험이 높다고 예측한 상위 K개 슬라이드 중
    실제 전이 슬라이드(label=1)의 비율.

    Args:
        slide_scores: (N_slides,) - 각 슬라이드의 전이 위험 점수 (높을수록 전이 의심)
        slide_labels: (N_slides,) - 실제 레이블 (1=전이, 0=정상)
        k:            상위 K개

    Returns:
        retrieval_rate: float in [0, 1]
    """
    if k > len(slide_scores):
        k = len(slide_scores)

    top_k_indices = np.argsort(slide_scores)[::-1][:k]
    top_k_labels = slide_labels[top_k_indices]

    return top_k_labels.sum() / k


def compute_sensitivity_at_fpr(
    slide_scores: np.ndarray,
    slide_labels: np.ndarray,
    target_fpr: float = 0.1,
) -> dict:
    """
    목표 FPR 이하에서 달성 가능한 최대 Sensitivity(TPR) 계산.

    Args:
        slide_scores: (N_slides,) - 전이 위험 점수
        slide_labels: (N_slides,) - 실제 레이블
        target_fpr:   허용 최대 위양성률 (e.g., 0.1 = 10%)

    Returns:
        dict with keys: sensitivity, threshold, auc_roc
    """
    fpr, tpr, thresholds = roc_curve(slide_labels, slide_scores)
    roc_auc = auc(fpr, tpr)

    # target_fpr 이하인 구간 중 TPR이 가장 높은 지점 선택
    valid_mask = fpr <= target_fpr
    if valid_mask.any():
        best_idx = np.where(valid_mask)[0][np.argmax(tpr[valid_mask])]
        sensitivity = float(tpr[best_idx])
        threshold = float(thresholds[best_idx])
    else:
        sensitivity = 0.0
        threshold = float(thresholds[-1])

    return {
        "sensitivity": sensitivity,
        "fpr": float(fpr[best_idx]) if valid_mask.any() else float(target_fpr),
        "threshold": threshold,
        "auc_roc": float(roc_auc),
    }


def compute_all_metrics(
    slide_scores: np.ndarray,
    slide_labels: np.ndarray,
    k: int = 20,
    target_fpr: float = 0.1,
) -> dict:
    """모든 지표를 한번에 계산."""
    top_k = compute_top_k_retrieval(slide_scores, slide_labels, k)
    sens = compute_sensitivity_at_fpr(slide_scores, slide_labels, target_fpr)

    return {
        f"top_{k}_retrieval_rate": top_k,
        **{f"sens@fpr{int(target_fpr*100)}_{name}": v for name, v in sens.items()},
    }


def compute_patch_metrics(
    patch_scores: np.ndarray,
    patch_preds: np.ndarray,
    patch_labels: np.ndarray,
) -> dict:
    """
    패치 레벨 분류 성능 지표.

    Args:
        patch_scores: (N,) softmax 양성 확률 [0, 1]
        patch_preds:  (N,) 이진 예측 (score >= 0.5 → 1)
        patch_labels: (N,) GT 라벨 (0 또는 1)

    Returns:
        dict: accuracy, auc_roc, f1, precision, recall
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )

    metrics = {
        "accuracy":  float(accuracy_score(patch_labels, patch_preds)),
        "f1":        float(f1_score(patch_labels, patch_preds, zero_division=0)),
        "precision": float(precision_score(patch_labels, patch_preds, zero_division=0)),
        "recall":    float(recall_score(patch_labels, patch_preds, zero_division=0)),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(patch_labels, patch_scores))
    except ValueError:
        metrics["auc_roc"] = float("nan")

    return metrics
