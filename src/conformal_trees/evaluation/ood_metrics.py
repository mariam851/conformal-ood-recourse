"""OOD Evaluation Metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc


def ood_auroc(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    """Computes standard AUROC where OOD is treated as the positive class (1)."""
    labels = np.concatenate([np.zeros_like(scores_id), np.ones_like(scores_ood)])
    all_scores = np.concatenate([scores_id, scores_ood])
    return float(roc_auc_score(labels, all_scores))


def fpr_at_tpr(scores_id: np.ndarray, scores_ood: np.ndarray, target_tpr: float = 0.95) -> float:
    """Computes False Positive Rate at target True Positive Rate (typically 95%)."""
    threshold = np.quantile(scores_ood, 1.0 - target_tpr)
    return float(np.mean(scores_id >= threshold))


def aupr_out(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    """Computes Area Under the Precision-Recall curve with OOD as the positive class."""
    labels = np.concatenate([np.zeros_like(scores_id), np.ones_like(scores_ood)])
    all_scores = np.concatenate([scores_id, scores_ood])
    precision, recall, _ = precision_recall_curve(labels, all_scores)
    return float(auc(recall, precision))


def ood_report(scores_id: np.ndarray, scores_ood: np.ndarray) -> dict[str, float]:
    """Summary report containing standard benchmark metrics."""
    return {
        "auroc": ood_auroc(scores_id, scores_ood),
        "fpr95": fpr_at_tpr(scores_id, scores_ood, target_tpr=0.95),
        "aupr_out": aupr_out(scores_id, scores_ood),
    }

