import numpy as np
import pandas as pd
import pytest

from conformal_trees.data.ood_scenarios import CovariateShift
from conformal_trees.evaluation import (
    classification_report,
    expected_calibration_error,
    fpr_at_tpr,
    ood_auroc,
    ood_report,
    shift_detectability,
)


def test_ece_zero_for_perfect_confident_predictions():
    y = np.array([0, 1, 1, 0])
    assert expected_calibration_error(y, np.array([0.0, 1.0, 1.0, 0.0])) == 0.0


def test_ece_detects_overconfidence():
    y = np.tile([0, 1], 500)                     # 50% accuracy when always predicting 1
    ece = expected_calibration_error(y, np.full(1000, 0.9))
    assert ece == pytest.approx(0.4, abs=1e-6)


def test_classification_report_keys_and_ranges():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.6 + rng.random(500) * 0.4, 0.01, 0.99)
    rep = classification_report(y, p)
    assert set(rep) == {"accuracy", "balanced_accuracy", "auroc", "log_loss", "brier", "ece"}
    assert rep["auroc"] > 0.9 and 0 <= rep["ece"] <= 1


def test_ood_auroc_perfect_chance_and_direction():
    assert ood_auroc([0, 1, 2], [3, 4, 5]) == 1.0
    assert ood_auroc([3, 4, 5], [0, 1, 2]) == 0.0          # higher score must mean "more OOD"
    s = np.arange(100.0)
    assert ood_auroc(s, s) == pytest.approx(0.5)


def test_fpr_at_tpr_known_example():
    s_ood = np.arange(1, 11, dtype=float)                  # 1..10: need threshold <= 1 for TPR 0.95
    s_id = np.array([0.0, 2.5, 3.5, 11.0])                 # 3 of 4 ID scores are >= 1
    assert fpr_at_tpr(s_id, s_ood, 0.95) == pytest.approx(0.75)
    assert fpr_at_tpr([0, 1, 2], [3, 4, 5], 0.95) == 0.0


def test_ood_report_keys():
    assert set(ood_report([0, 1], [2, 3])) == {"auroc", "fpr95", "aupr_out"}


def test_shift_detectability_chance_and_saturation(shift_frame):
    numeric = ["a", "b", "c"]
    same = shift_detectability(shift_frame, shift_frame.copy(), seed=0)
    assert abs(same - 0.5) < 0.05
    strong = CovariateShift("mean_shift", 3.0).fit(shift_frame, numeric).apply(shift_frame)
    assert shift_detectability(shift_frame, strong, seed=0) > 0.95


def test_small_shift_on_integer_features_is_not_trivially_detectable():
    """Regression test: fractional-part artifacts must not make small shifts look separable."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"n": rng.integers(0, 100, 4000), "m": rng.integers(0, 50, 4000)})
    tiny = CovariateShift("gaussian_noise", 0.05).fit(X, ["n", "m"]).apply(X, seed=0)
    assert shift_detectability(X, tiny, seed=0) < 0.8


def test_spike_features_are_what_made_scenarios_trivial():
    """Regression test (Adult-like): zero-inflated features alone give near-perfect separation."""
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame(
        {
            "age": rng.integers(17, 90, n),
            "gain": np.where(rng.random(n) < 0.92, 0, rng.integers(1, 99999, n)),
        }
    )
    kw = dict(severity=0.25)
    dense = CovariateShift("gaussian_noise", **kw).fit(X, ["age", "gain"]).apply(X, seed=0)
    spiky = CovariateShift("gaussian_noise", features="all_numeric", **kw).fit(X, ["age", "gain"]).apply(X, seed=0)
    assert shift_detectability(X, dense, seed=0) < 0.8
    assert shift_detectability(X, spiky, seed=0) > 0.95
    