"""Small statistics helpers for paired comparisons."""
from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    a, b, *, rng: np.random.Generator, n_boot: int = 2000, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean of (a - b) over paired items with a percentile bootstrap CI: (mean, low, high)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(d.mean()), float(lo), float(hi)
