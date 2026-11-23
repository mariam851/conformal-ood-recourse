import numpy as np
import pandas as pd
import pytest

from conformal_trees.conformal import conformal_pvalues, empirical_fpr
from conformal_trees.data import TabularEncoder
from conformal_trees.models import TreeEmbedder
from conformal_trees.signals import reference_score


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "k": pd.Categorical(rng.choice(["x", "y", "z"], n)),
        }
    )
    y = ((X["a"] + X["b"] + (X["k"] == "x") + rng.normal(size=n)) > 0.5).astype(int).to_numpy()
    return X, y


def test_encoder_handles_nan_unseen_categories_and_is_fit_on_train_only(frame):
    X, _ = frame
    enc = TabularEncoder(["a", "b"], ["k"]).fit(X.iloc[:100])
    new = pd.DataFrame({"a": [np.nan], "b": [0.0], "k": pd.Categorical(["unseen"])})
    out = enc.transform(new)
    assert out.shape == (1, 3)
    assert out[0, 0] == pytest.approx(X["a"].iloc[:100].median())     # NaN -> TRAIN median
    assert out[0, 2] == -1.0                                           # unseen category


def test_embedder_shape_dtype_and_determinism(frame):
    X, y = frame
    mk = lambda seed: TreeEmbedder(["a", "b"], ["k"], n_estimators=20, n_jobs=1, seed=seed).fit(X.iloc[:3000], y[:3000])
    e1, e2, e3 = mk(1), mk(1), mk(2)
    t1, t2, t3 = (e.transform(X.iloc[3000:3500]) for e in (e1, e2, e3))
    assert t1.shape == (500, 20) and np.issubdtype(t1.dtype, np.integer)
    assert np.array_equal(t1, t2)
    assert not np.array_equal(t1, t3)


def _id_fpr_once(rep, alphas):
    rng = np.random.default_rng(1000 + rep)
    n = 6000
    X = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "k": pd.Categorical(rng.choice(["x", "y", "z"], n))}
    )
    y = ((X["a"] + X["b"] + (X["k"] == "x") + rng.normal(size=n)) > 0.5).astype(int).to_numpy()
    emb = TreeEmbedder(["a", "b"], ["k"], n_estimators=20, n_jobs=1, seed=rep).fit(X.iloc[:2000], y[:2000])
    ref, cal, test = (emb.transform(X.iloc[a:b]) for a, b in [(2000, 3000), (3000, 4500), (4500, 6000)])
    p = conformal_pvalues(reference_score(cal, ref), reference_score(test, ref))
    return [empirical_fpr(p, a) for a in alphas]


def test_reference_based_conformal_ood_score_is_valid_on_id_data():
    """End-to-end validity: embedder -> reference score -> split-conformal p-values.

    The guarantee (FPR <= alpha) holds on average over calibration draws; a single run fluctuates
    by ~sqrt(alpha(1-alpha)/n_cal) around its mean (Bates et al.), so we average over replicates.
    With fully grown trees the scores contain many ties, and ties count against the new point, so
    the procedure is valid but conservative: 250 replicates gave mean FPR 0.0457 at alpha=0.05
    and 0.1927 at alpha=0.2 (never above alpha).
    """
    alphas = (0.05, 0.2)
    fprs = np.array([_id_fpr_once(rep, alphas) for rep in range(12)])
    assert fprs[:, 0].mean() < 0.05 + 0.012
    assert fprs[:, 1].mean() < 0.20 + 0.02
    assert fprs[:, 0].mean() > 0.05 - 0.02          # and it must not be absurdly conservative either
    