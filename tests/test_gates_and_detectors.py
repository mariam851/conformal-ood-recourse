import numpy as np
import pandas as pd
import pytest

from conformal_trees.recourse import AllOfGate, OODGate
from conformal_trees.signals import GMMScore, OCSVMScore


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)

    def make(n, loc=0.0):
        return pd.DataFrame(
            {
                "a": rng.normal(loc, 1, n),
                "b": rng.normal(loc, 1, n),
                "k": pd.Categorical(rng.choice(["x", "y"], n), categories=["x", "y"]),
            }
        )

    return make(3000), make(500), make(500, loc=6.0)


@pytest.mark.parametrize("cls", [GMMScore, OCSVMScore])
def test_held_out_detectors_score_far_rows_higher(cls, frames):
    train, id_rows, far_rows = frames
    det = cls(["a", "b"], ["k"]).fit(train)
    s_id, s_far = det.score(id_rows), det.score(far_rows)
    assert s_id.shape == (500,) and np.isfinite(s_id).all()
    assert s_far.mean() > s_id.mean()
    assert np.mean(s_far > np.quantile(s_id, 0.95)) > 0.9


@pytest.mark.parametrize("cls", [GMMScore, OCSVMScore])
def test_held_out_detectors_are_deterministic(cls, frames):
    train, id_rows, _ = frames
    a = cls(["a", "b"], ["k"], seed=3).fit(train).score(id_rows)
    b = cls(["a", "b"], ["k"], seed=3).fit(train).score(id_rows)
    assert np.array_equal(a, b)


def test_ocsvm_subsamples_large_training_sets(frames):
    train, id_rows, _ = frames
    det = OCSVMScore(["a", "b"], ["k"], max_fit_rows=400).fit(train)
    assert det.model.support_vectors_.shape[0] <= 400
    assert det.score(id_rows).shape == (500,)


def test_all_of_gate_is_a_conjunction_and_sums_violations():
    cal = np.abs(np.random.default_rng(5).normal(size=6000))
    g1 = OODGate(lambda X: X["s"].to_numpy(float), cal, alpha=0.05, delta=0.01)
    g2 = OODGate(lambda X: X["t"].to_numpy(float), cal, alpha=0.05, delta=0.01)
    both = AllOfGate([g1, g2])
    X = pd.DataFrame({"s": [0.1, 9.0, 0.1, 9.0], "t": [0.1, 0.1, 9.0, 9.0]})
    assert both.passes(X).tolist() == [True, False, False, False]
    assert np.allclose(both.violation(X), g1.violation(X) + g2.violation(X))
    assert both.violation(X)[0] == 0.0 and both.violation(X)[3] > both.violation(X)[1]
    with pytest.raises(ValueError):
        AllOfGate([])


def test_all_of_gate_is_never_more_permissive_than_its_parts():
    rng = np.random.default_rng(7)
    cal = np.abs(rng.normal(size=6000))
    g1 = OODGate(lambda X: X["s"].to_numpy(float), cal)
    g2 = OODGate(lambda X: X["t"].to_numpy(float), cal)
    X = pd.DataFrame({"s": np.abs(rng.normal(size=5000)), "t": np.abs(rng.normal(size=5000))})
    both = AllOfGate([g1, g2]).passes(X)
    assert both.mean() <= min(g1.passes(X).mean(), g2.passes(X).mean())
    assert not (both & ~g1.passes(X)).any() and not (both & ~g2.passes(X)).any()
    