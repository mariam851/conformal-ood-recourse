"""OOD gates for recourse: a counterfactual must not be flagged by a conformal OOD test.

``OODGate`` wraps ONE per-row OOD score (higher = more OOD) calibrated on ID calibration rows. With the
calibration-conditional rule (fpr_bound.py), a point is flagged when its score exceeds the k-th largest
calibration score, and with probability >= 1 - delta over the calibration set this rule flags at most a
fraction ``alpha`` of in-distribution points.

``AllOfGate`` is a conjunction of gates (a row passes only if every gate passes), used to test whether a gate built
on several detectors also helps a detector it was not built on.

What a PASS means: "this counterfactual is not rejected by the test(s) with FPR <= alpha". It does NOT mean the
counterfactual is in-distribution or realistic: the counterfactual is selected by a search that optimises
against the gate, so it is not exchangeable with the calibration rows and no coverage claim holds for it
(the exchangeability problem noted by CONFEX). Realism is measured separately with detectors that are
independent of the gate (evaluation/recourse_metrics.py).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from conformal_trees.conformal.fpr_bound import conditional_threshold


class OODGate:
    def __init__(
        self,
        score_fn: Callable[[pd.DataFrame], np.ndarray],
        cal_scores,
        alpha: float = 0.05,
        delta: float = 0.01,
    ) -> None:
        cal = np.sort(np.asarray(cal_scores, dtype=float))
        n = len(cal)
        self.score_fn = score_fn
        self.k, _ = conditional_threshold(n, alpha, delta)
        self.score_threshold = float(cal[n - self.k]) if self.k > 0 else np.inf   # k-th largest
        self.scale = float(cal.std()) or 1.0
        self.alpha, self.delta, self.n_cal = alpha, delta, n

    def violation(self, X: pd.DataFrame) -> np.ndarray:
        """0 when the row passes; otherwise how far its score is above the threshold (in calibration sds)."""
        s = np.asarray(self.score_fn(X), dtype=float)
        return np.maximum(0.0, (s - self.score_threshold) / self.scale)

    def passes(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.score_fn(X), dtype=float) <= self.score_threshold


class AllOfGate:
    """Conjunction of gates: a row passes only if EVERY gate passes. The violation is the sum of the violations,
    so the search is pushed to satisfy all detectors at once."""

    def __init__(self, gates) -> None:
        self.gates = list(gates)
        if not self.gates:
            raise ValueError("AllOfGate needs at least one gate.")

    def violation(self, X: pd.DataFrame) -> np.ndarray:
        return np.sum([g.violation(X) for g in self.gates], axis=0)

    def passes(self, X: pd.DataFrame) -> np.ndarray:
        return np.all([g.passes(X) for g in self.gates], axis=0)
    