"""Calibration-conditional FPR bound for conformal OOD detection.

The marginal guarantee (P(p <= alpha) <= alpha) is an average over calibration sets. One
calibration set of size n has a realised FPR that fluctuates around alpha by about
sqrt(alpha (1 - alpha) / n) (Bates et al.), so benchmarks can look better than they are
(Novello et al., 2024).

Exact result (continuous scores). Flag a new point when its upper conformal p-value is
<= k / (n + 1), i.e. when its score exceeds the k-th largest calibration score. Conditionally
on the calibration set, the true FPR is distributed as Beta(k, n - k + 1). Hence with
probability >= 1 - delta over the calibration set:

    FPR <= FPR+(n, k, delta) = Beta(k, n - k + 1).ppf(1 - delta).

Two-sided rule (flag if min(p_hi, p_lo) <= k / (n + 1)): each tail is Beta(k, n - k + 1);
a union bound with delta / 2 per tail gives FPR <= 2 * Beta(k, n - k + 1).ppf(1 - delta / 2).
With ties the guarantee stays valid (conservative).

``conditional_k`` picks the largest k whose bound is <= alpha: the most permissive rule that still
meets the target FPR with confidence 1 - delta. The price is power compared with the marginal rule.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta


def marginal_rank(n: int, alpha: float) -> int:
    """Rank cutoff of the marginal rule: flag when p <= k/(n+1), k = floor(alpha (n+1))."""
    return int(np.floor(alpha * (n + 1)))


def fpr_upper_bound(n: int, k: int, delta: float, *, two_sided: bool = False) -> float:
    """With probability >= 1 - delta over calibration sets, FPR of the rank-k rule is <= this."""
    if k <= 0:
        return 0.0
    if k > n:
        return 1.0
    d = delta / 2.0 if two_sided else delta
    b = float(beta.ppf(1.0 - d, k, n - k + 1))
    return min(1.0, 2.0 * b) if two_sided else b


def conditional_k(n: int, alpha: float, delta: float, *, two_sided: bool = False) -> int:
    """Largest k with fpr_upper_bound(n, k, delta) <= alpha (0 = never flag, n too small)."""
    lo, hi = 0, n
    while lo < hi:                                    # the bound is increasing in k
        mid = (lo + hi + 1) // 2
        if fpr_upper_bound(n, mid, delta, two_sided=two_sided) <= alpha:
            lo = mid
        else:
            hi = mid - 1
    return lo


def conditional_threshold(
    n: int, alpha: float, delta: float, *, two_sided: bool = False
) -> tuple[int, float]:
    """(k, p-value threshold) of the calibration-conditional rule.

    Use the threshold with the matching p-values: ``p <= threshold`` where p comes from
    ``conformal_pvalues(..., sided="upper")`` (one-sided) or ``sided="two"`` (two-sided; its
    p-value is 2 * min(p_hi, p_lo), so the threshold is 2k/(n+1)). Do NOT use smoothed p-values here.
    """
    k = conditional_k(n, alpha, delta, two_sided=two_sided)
    return k, (2.0 if two_sided else 1.0) * k / (n + 1.0)
