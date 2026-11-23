"""Recourse metrics. Realism is measured with detectors INDEPENDENT of the gate used in the search."""
from __future__ import annotations

import numpy as np
import pandas as pd

from conformal_trees.conformal.calibration import conformal_pvalues


def flag_rate(cal_scores, scores, alpha: float = 0.05) -> float:
    """Share of rows flagged as OOD by a split-conformal one-sided test at level alpha.

    On in-distribution rows this is about alpha, so it is a calibrated 'realism' reading:
    a counterfactual set with a flag rate far above alpha looks out-of-distribution to that detector.
    """
    return float(np.mean(conformal_pvalues(cal_scores, scores) <= alpha))


def robust_validity(models, X_cf: pd.DataFrame, target_prob: float = 0.5) -> np.ndarray:
    """Per counterfactual: share of alternative models (other seeds) that still give P(y=1) >= target."""
    ok = np.column_stack([m.predict_proba(X_cf)[:, 1] >= target_prob for m in models])
    return ok.mean(axis=1)
