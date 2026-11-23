"""Metrics for the base classifier. ``p_pos`` is always P(y = 1)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(y_true, p_pos, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins (confidence lies in [0.5, 1])."""
    y = np.asarray(y_true)
    p = np.asarray(p_pos, dtype=float)
    pred = (p >= 0.5).astype(int)
    confidence = np.where(pred == 1, p, 1.0 - p)
    correct = (pred == y).astype(float)

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def classification_report(y_true, p_pos) -> dict[str, float]:
    y = np.asarray(y_true)
    p = np.asarray(p_pos, dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }
