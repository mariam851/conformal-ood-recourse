"""Synthetic covariate-shift OOD scenarios for tabular data.

Design rules (each one protects the validity of the experiments):
  * Shift statistics (per-feature std) are measured on a REFERENCE frame, which
    must be the ``train`` split. They are never computed from the rows being shifted.
  * OOD sets are built by shifting rows of the ``test`` split. The ID set and the
    OOD set are therefore *paired* (same underlying rows), so any difference in a
    detector's behaviour is caused by the shift and not by sampling noise.
  * Severity is expressed in units of the per-feature std, so scenarios are
    comparable across features and datasets.
  * Spike-at-a-value features are excluded by default (``dense_numeric``). On Adult,
    capital-gain / capital-loss are 92-95% zeros; a shift in std units turns them into
    impossible values (negative gains, non-zero for everyone) and makes every scenario
    trivially separable, regardless of the other features. ``all_numeric`` remains
    available to measure that effect explicitly.
  * Granularity is preserved: features that are integer-valued in the reference
    frame stay integer-valued after the shift. Without this, shifted rows carry
    fractional values and a detector can separate ID from OOD by the fractional
    part alone.
  * No clipping: shifted values may leave the observed range. That is the point of
    an OOD scenario, but it also means these points can be implausible. Natural
    shifts (e.g. TableShift/folktables) are a separate scenario family.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from conformal_trees.utils.seed import derive_seed, make_rng

SHIFT_KINDS = ("mean_shift", "gaussian_noise")


def _mode_share(col: pd.Series) -> float:
    """Fraction of (non-missing) rows that take the most common value."""
    counts = col.dropna().value_counts(normalize=True)
    return float(counts.iloc[0]) if len(counts) else 1.0


def _is_integer_valued(col: pd.Series) -> bool:
    v = col.dropna().to_numpy(dtype=float)
    return bool(len(v)) and bool(np.all(v == np.round(v)))


class CovariateShift:
    """Shift numeric features by ``severity`` standard deviations.

    mean_shift     : x -> x + severity * std
    gaussian_noise : x -> x + N(0, (severity * std)^2)

    Integer-valued features are rounded back to integers after the shift.
    """

    def __init__(
        self,
        kind: str,
        severity: float,
        features: str | Sequence[str] = "dense_numeric",
        max_mode_share: float = 0.75,
    ) -> None:
        if kind not in SHIFT_KINDS:
            raise ValueError(f"Unknown shift kind {kind!r}; expected one of {SHIFT_KINDS}")
        if not severity > 0:
            raise ValueError(f"severity must be > 0, got {severity}")
        self.kind = kind
        self.severity = float(severity)
        self.features = features
        self.max_mode_share = float(max_mode_share)
        self.scale_: dict[str, float] | None = None   # feature -> std on the reference frame
        self.skipped_: list[str] = []                 # features with zero/undefined std
        self.integer_features_: list[str] = []        # integer-valued on the reference frame
        self.excluded_spiky_: list[str] = []          # dropped by the dense_numeric rule

    @property
    def name(self) -> str:
        return f"{self.kind}@{self.severity:g}"

    def fit(self, X_ref: pd.DataFrame, numeric: Sequence[str]) -> "CovariateShift":
        """Measure per-feature std on the reference frame (use the train split)."""
        numeric = list(numeric)
        self.excluded_spiky_ = []
        if self.features == "all_numeric":
            cols = numeric
        elif self.features == "dense_numeric":
            shares = {c: _mode_share(X_ref[c]) for c in numeric}
            self.excluded_spiky_ = [c for c in numeric if shares[c] > self.max_mode_share]
            cols = [c for c in numeric if c not in self.excluded_spiky_]
        else:
            cols = list(self.features)
        not_numeric = [c for c in cols if c not in numeric]
        if not_numeric:
            raise ValueError(f"Shift features must be numeric columns; got {not_numeric}")

        std = X_ref[cols].astype(float).std()
        self.skipped_ = [c for c in cols if not np.isfinite(std[c]) or std[c] == 0.0]
        self.scale_ = {c: float(std[c]) for c in cols if c not in self.skipped_}
        if not self.scale_:
            raise ValueError("No shiftable feature: all selected features have zero variance.")
        self.integer_features_ = [c for c in self.scale_ if _is_integer_valued(X_ref[c])]
        return self

    def apply(self, X: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        """Return a shifted copy of ``X`` (the input is never modified)."""
        if self.scale_ is None:
            raise RuntimeError("Call fit() on the train split before apply().")
        out = X.copy()
        rng = make_rng(derive_seed(seed, "ood", self.name))
        for col, sigma in self.scale_.items():
            values = out[col].astype(float).to_numpy()
            if self.kind == "mean_shift":
                values = values + self.severity * sigma
            else:
                values = values + rng.normal(0.0, self.severity * sigma, size=len(values))
            if col in self.integer_features_:
                values = np.round(values)
            out[col] = values
        return out

    def describe(self) -> pd.DataFrame:
        """Per-feature magnitude of the perturbation (offset, or noise std)."""
        if self.scale_ is None:
            raise RuntimeError("Call fit() first.")
        return pd.DataFrame(
            {
                "train_std": self.scale_,
                "delta": {c: self.severity * s for c, s in self.scale_.items()},
            }
        ).rename_axis("feature")


def build_covariate_shifts(
    cfg: dict[str, Any], X_ref: pd.DataFrame, numeric: Sequence[str]
) -> list[CovariateShift]:
    """Build and fit the full kinds x severities grid from configs/ood_scenarios.yaml."""
    c = cfg["covariate_shift"]
    features = c.get("features", "dense_numeric")
    max_mode_share = c.get("max_mode_share", 0.75)
    return [
        CovariateShift(kind, severity, features, max_mode_share).fit(X_ref, numeric)
        for kind in c["kinds"]
        for severity in c["severities"]
    ]

def mix_batch(
    X_id: pd.DataFrame,
    X_ood: pd.DataFrame,
    ood_fraction: float = 0.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Mixes ID and paired OOD rows at the exact same row positions."""
    if not (0.0 <= ood_fraction <= 1.0):
        raise ValueError(f"ood_fraction must be in [0, 1], got {ood_fraction}")

    n = len(X_id)
    if len(X_ood) != n:
        raise ValueError(f"X_id and X_ood must have identical lengths, got {n} vs {len(X_ood)}")

    n_ood = int(round(n * ood_fraction))
    rng = np.random.default_rng(seed)

    is_ood = np.zeros(n, dtype=bool)
    if n_ood > 0:
        ood_idx = rng.choice(n, size=n_ood, replace=False)
        is_ood[ood_idx] = True

    batch = X_id.copy()
    if is_ood.any():
        batch.loc[is_ood] = X_ood.loc[is_ood]

    return batch, pd.Series(is_ood, index=X_id.index)
