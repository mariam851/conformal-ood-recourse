import numpy as np
import pandas as pd
import pytest

from conformal_trees.signals import IsolationForestScore, KNNScore, LOFScore


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


@pytest.mark.parametrize("cls", [IsolationForestScore, LOFScore, KNNScore])
def test_far_rows_score_higher_than_id_rows(cls, frames):
    train, id_rows, far_rows = frames
    det = cls(["a", "b"], ["k"], n_jobs=1).fit(train)
    s_id, s_far = det.score(id_rows), det.score(far_rows)
    assert s_id.shape == (500,) and np.isfinite(s_id).all()
    assert s_far.mean() > s_id.mean()
    assert np.mean(s_far > np.quantile(s_id, 0.95)) > 0.9        # far rows are flagged by a 5% ID threshold


def test_knn_score_is_a_mean_distance_and_deterministic(frames):
    train, id_rows, _ = frames
    a = KNNScore(["a", "b"], ["k"], k=5, n_jobs=1).fit(train).score(id_rows)
    b = KNNScore(["a", "b"], ["k"], k=5, n_jobs=1).fit(train).score(id_rows)
    assert np.array_equal(a, b) and (a >= 0).all()
    