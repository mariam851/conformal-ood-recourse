"""Split-conformal p-values for OOD detection (higher score = more OOD).

Let a = #{cal > s}, b = #{cal == s}, c = #{cal < s} for a new score s and n calibration scores.

Plain p-values (ties count against the new point, so validity is kept but can be conservative):
    upper:  p_hi = (1 + a + b) / (n + 1)        small when s is unusually HIGH
    lower:  p_lo = (1 + c + b) / (n + 1)        small when s is unusually LOW
    two:    p    = min(1, 2 * min(p_hi, p_lo))  small when s is unusual in EITHER direction

Smoothed (randomised) p-values break ties with U ~ Uniform(0, 1):
    p_hi = (a + U * (b + 1)) / (n + 1),   p_lo = 1 - p_hi
Under exchangeability a smoothed p-value is exactly Uniform(0, 1), so "flag when p <= alpha"
has FPR = alpha (not just <= alpha). This removes the conservativeness caused by tied scores.

Guarantees (marginal over the draw of the calibration set), for exchangeable ID scores:
    P(p_hi <= alpha) <= alpha,   P(p_lo <= alpha) <= alpha,
    P(p_two <= alpha) <= alpha   (union bound over the two tails at alpha / 2 each).

Why two-sided: the direction in which an OOD score moves under shift is NOT stable (it depends
on the shift kind and on tree depth, see docs/decisions.md D10), and it cannot be chosen without
OOD labels. A two-sided test needs no direction, at the price of some power when the direction is known.

The calibration-conditional guarantee (Bates et al.; Novello et al.) is in ``fpr_bound.py``.
"""
from __future__ import annotations

import numpy as np

_SIDES = ("upper", "lower", "two")


def _counts(cal_scores, test_scores):
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    s = np.asarray(test_scores, dtype=float)
    n = len(cal)
    below = np.searchsorted(cal, s, side="left")          # c = #{cal <  s}
    at_most = np.searchsorted(cal, s, side="right")       # #{cal <= s}
    return n, n - at_most, at_most - below, below          # n, a, b, c


def conformal_pvalues(
    cal_scores,
    test_scores,
    *,
    sided: str = "upper",
    smoothed: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if sided not in _SIDES:
        raise ValueError(f"sided must be one of {_SIDES}, got {sided!r}")
    n, a, b, c = _counts(cal_scores, test_scores)
    if smoothed:
        if rng is None:
            raise ValueError("smoothed p-values need an explicit rng (use make_rng(derive_seed(...))).")
        p_hi = (a + rng.random(len(a)) * (b + 1)) / (n + 1.0)
        p_lo = 1.0 - p_hi
    else:
        p_hi = (1.0 + a + b) / (n + 1.0)
        p_lo = (1.0 + c + b) / (n + 1.0)
    if sided == "upper":
        return p_hi
    if sided == "lower":
        return p_lo
    return np.minimum(1.0, 2.0 * np.minimum(p_hi, p_lo))


def empirical_fpr(p_values_id, alpha: float) -> float:
    """Fraction of ID points flagged as OOD at level alpha (should be <= alpha)."""
    return float(np.mean(np.asarray(p_values_id) <= alpha))
