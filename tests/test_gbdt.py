import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from conformal_trees.models import GBDTClassifier

PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.3,
    "num_leaves": 15,
    "subsample": 0.8,
    "subsample_freq": 1,
    "n_jobs": 1,
}


@pytest.fixture
def task():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cat": pd.Categorical(rng.choice(list("abc"), n)),
        }
    )
    logit = 2 * X["x1"] - X["x2"] + (X["cat"] == "a") * 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def _fit(task, seed=0):
    X, y = task
    return GBDTClassifier(PARAMS, early_stopping_rounds=10, val_fraction=0.1, seed=seed).fit(X, y)


def test_learns_the_signal_and_returns_valid_probabilities(task):
    X, y = task
    model = _fit(task)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert roc_auc_score(y, proba[:, 1]) > 0.8


def test_early_stopping_uses_fewer_trees_than_the_cap(task):
    model = _fit(task)
    assert 0 < model.best_iteration_ < PARAMS["n_estimators"]


def test_same_seed_is_bit_for_bit_reproducible(task):
    X, _ = task
    a, b = _fit(task, seed=5), _fit(task, seed=5)
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))


def test_different_seed_changes_the_model(task):
    X, _ = task
    a, b = _fit(task, seed=1), _fit(task, seed=2)
    assert not np.array_equal(a.predict_proba(X), b.predict_proba(X))


def test_column_mismatch_is_rejected(task):
    X, _ = task
    model = _fit(task)
    with pytest.raises(ValueError, match="Columns differ"):
        model.predict_proba(X[["x2", "x1", "cat"]])      # same names, wrong order
    with pytest.raises(RuntimeError):
        GBDTClassifier(PARAMS).predict_proba(X)          # not fitted


def test_feature_importance_ranks_informative_features_first(task):
    imp = _fit(task).feature_importance()
    assert set(imp.index) == {"x1", "x2", "cat"}
    assert imp.index[0] == "x1"


def test_from_config_and_bad_val_fraction():
    cfg = {"early_stopping_rounds": 7, "val_fraction": 0.2, "lightgbm": {"n_estimators": 10}}
    m = GBDTClassifier.from_config(cfg, seed=3)
    assert (m.early_stopping_rounds, m.val_fraction, m.seed) == (7, 0.2, 3)
    with pytest.raises(ValueError):
        GBDTClassifier({}, val_fraction=1.0)
    