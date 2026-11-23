import numpy as np
import pytest

from conformal_trees.conformal import conformal_pvalues, empirical_fpr


def test_known_values():
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    p = conformal_pvalues(cal, [0.5, 2.5, 4.0, 10.0])
    assert p.tolist() == pytest.approx([1.0, 3 / 5, 2 / 5, 1 / 5])   # (1 + #{cal >= s}) / (n + 1)


def test_ties_count_against_the_new_point():
    cal = np.array([1.0, 1.0, 1.0])
    assert conformal_pvalues(cal, [1.0])[0] == pytest.approx(1.0)


def test_marginal_validity_monte_carlo():
    """P(p <= alpha) <= alpha for exchangeable scores (here: exactly 10/200 = 0.05 for n=199)."""
    rng = np.random.default_rng(0)
    n, trials, alpha = 199, 20000, 0.05
    hits = 0
    for _ in range(trials):
        cal = rng.normal(size=n)
        hits += conformal_pvalues(cal, rng.normal(size=1))[0] <= alpha
    assert 0.04 < hits / trials < 0.06


def test_empirical_fpr():
    assert empirical_fpr([0.01, 0.2, 0.04, 0.9], 0.05) == 0.5


def test_lower_and_two_sided_known_values():
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    lower = conformal_pvalues(cal, [0.5, 2.5, 10.0], sided="lower")
    assert lower.tolist() == pytest.approx([1 / 5, 3 / 5, 1.0])
    two = conformal_pvalues(cal, [0.5, 2.5, 10.0], sided="two")
    assert two.tolist() == pytest.approx([2 / 5, 1.0, 2 / 5])        # min(1, 2 * min(p_hi, p_lo))


def test_invalid_arguments():
    with pytest.raises(ValueError):
        conformal_pvalues([1.0, 2.0], [1.5], sided="both")
    with pytest.raises(ValueError):
        conformal_pvalues([1.0, 2.0], [1.5], smoothed=True)          # smoothed needs an rng


def test_smoothed_upper_and_lower_pvalues_are_complementary():
    rng = np.random.default_rng(0)
    cal = rng.integers(0, 5, size=50).astype(float)                  # many ties
    test = rng.integers(0, 5, size=200).astype(float)
    hi = conformal_pvalues(cal, test, smoothed=True, rng=np.random.default_rng(1))
    lo = conformal_pvalues(cal, test, sided="lower", smoothed=True, rng=np.random.default_rng(1))
    assert np.allclose(hi + lo, 1.0)


def test_two_sided_marginal_validity_monte_carlo():
    rng = np.random.default_rng(1)
    n, trials, alpha = 199, 20000, 0.1
    hits = sum(
        conformal_pvalues(rng.normal(size=n), rng.normal(size=1), sided="two")[0] <= alpha
        for _ in range(trials)
    )
    assert hits / trials <= alpha + 0.01


def test_smoothing_removes_conservativeness_caused_by_ties():
    """Discrete scores: plain p-values under-cover (FPR < alpha); smoothed ones hit alpha."""
    rng = np.random.default_rng(2)
    n, trials, alpha = 99, 6000, 0.05
    plain = smooth = 0
    for _ in range(trials):
        cal = rng.integers(0, 10, size=n).astype(float)
        s = rng.integers(0, 10, size=1).astype(float)
        plain += conformal_pvalues(cal, s)[0] <= alpha
        smooth += conformal_pvalues(cal, s, smoothed=True, rng=rng)[0] <= alpha
    assert plain / trials < 0.035                                    # clearly conservative
    assert abs(smooth / trials - alpha) < 0.012                      # (nearly) exact


def test_two_sided_detects_an_inverted_signal_that_one_sided_misses():
    rng = np.random.default_rng(3)
    cal = rng.normal(size=2000)
    shifted_down = rng.normal(loc=-2.0, size=2000)                   # score goes DOWN under shift
    upper = empirical_fpr(conformal_pvalues(cal, shifted_down), 0.05)
    two = empirical_fpr(conformal_pvalues(cal, shifted_down, sided="two"), 0.05)
    assert upper < 0.01 and two > 0.5
    