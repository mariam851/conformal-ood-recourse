import numpy as np
import pytest
from scipy.stats import norm

from conformal_trees.conformal import conditional_k, conditional_threshold, fpr_upper_bound, marginal_rank


def _coverage(n, k, delta, two_sided, draws=4000, seed=0):
    """Share of calibration draws whose TRUE conditional FPR is <= the bound (N(0,1) scores)."""
    rng = np.random.default_rng(seed)
    bound = fpr_upper_bound(n, k, delta, two_sided=two_sided)
    ok = 0
    for _ in range(draws):
        cal = np.sort(rng.normal(size=n))
        hi, lo = cal[n - k], cal[k - 1]                              # k-th largest / k-th smallest
        fpr = norm.sf(hi) + (norm.cdf(lo) if two_sided else 0.0)
        ok += fpr <= bound
    return ok / draws


def test_one_sided_bound_has_exact_coverage():
    n, delta = 200, 0.1
    k = marginal_rank(n, 0.05)
    assert k == 10
    assert abs(_coverage(n, k, delta, two_sided=False) - (1 - delta)) < 0.025


def test_two_sided_bound_is_valid_and_not_absurdly_loose():
    n, delta = 200, 0.1
    k = marginal_rank(n, 0.1) // 2
    cov = _coverage(n, k, delta, two_sided=True)
    assert cov >= 1 - delta - 0.02
    assert cov < 0.999


def test_conditional_k_is_the_largest_k_meeting_the_target():
    for n, alpha, delta, two in [(5000, 0.05, 0.01, False), (5000, 0.05, 0.01, True), (800, 0.1, 0.05, False)]:
        k = conditional_k(n, alpha, delta, two_sided=two)
        assert fpr_upper_bound(n, k, delta, two_sided=two) <= alpha
        assert fpr_upper_bound(n, k + 1, delta, two_sided=two) > alpha
        assert k <= marginal_rank(n, alpha)                          # never more permissive than the marginal rule


def test_threshold_is_more_conservative_than_marginal_and_converges_with_n():
    small = conditional_threshold(500, 0.05, 0.01)[1]
    large = conditional_threshold(50000, 0.05, 0.01)[1]
    assert small < large < 0.05
    assert 0.05 - large < 0.005                                      # n large: almost no power lost


def test_two_sided_threshold_uses_doubled_scale():
    k1, t1 = conditional_threshold(5000, 0.05, 0.01)
    k2, t2 = conditional_threshold(5000, 0.05, 0.01, two_sided=True)
    assert t1 == pytest.approx(k1 / 5001) and t2 == pytest.approx(2 * k2 / 5001)
    assert k2 < k1                                                   # two tails + delta/2 -> stricter per tail


def test_tiny_calibration_set_never_flags():
    k, thr = conditional_threshold(20, 0.05, 0.01)
    assert (k, thr) == (0, 0.0)
    assert fpr_upper_bound(20, 0, 0.01) == 0.0
    