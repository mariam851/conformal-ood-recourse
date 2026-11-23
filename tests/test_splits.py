import numpy as np
import pytest

from conformal_trees.data.splits import SPLIT_NAMES, make_splits

FRACTIONS = {"train": 0.5, "calib_fit": 0.1, "calib": 0.2, "test": 0.2}


def test_sizes_match_fractions(toy_xy):
    X, y = toy_xy
    s = make_splits(X, y, FRACTIONS, seed=0)
    for name, split in s:
        assert abs(len(split) / len(y) - FRACTIONS[name]) < 0.01


def test_disjoint_and_cover_everything(toy_xy):
    X, y = toy_xy
    s = make_splits(X, y, FRACTIONS, seed=0)
    all_idx = np.concatenate([split.idx for _, split in s])
    assert len(all_idx) == len(y)
    assert len(np.unique(all_idx)) == len(y)


def test_calib_and_calib_fit_are_disjoint(toy_xy):
    X, y = toy_xy
    s = make_splits(X, y, FRACTIONS, seed=0)
    assert set(s.calib.idx).isdisjoint(set(s.calib_fit.idx))


def test_stratified(toy_xy):
    X, y = toy_xy
    s = make_splits(X, y, FRACTIONS, seed=0)
    for _, split in s:
        assert abs(split.y.mean() - y.mean()) < 0.01


def test_rows_and_labels_stay_aligned(toy_xy):
    X, y = toy_xy
    s = make_splits(X, y, FRACTIONS, seed=0)
    for _, split in s:
        assert np.array_equal(split.y, y[split.idx])
        assert np.allclose(split.X["a"].to_numpy(), X["a"].to_numpy()[split.idx])


def test_deterministic_for_same_seed(toy_xy):
    X, y = toy_xy
    a = make_splits(X, y, FRACTIONS, seed=7)
    b = make_splits(X, y, FRACTIONS, seed=7)
    for name in SPLIT_NAMES:
        assert np.array_equal(getattr(a, name).idx, getattr(b, name).idx)


def test_different_seed_gives_different_split(toy_xy):
    X, y = toy_xy
    a = make_splits(X, y, FRACTIONS, seed=1)
    b = make_splits(X, y, FRACTIONS, seed=2)
    assert not np.array_equal(a.test.idx, b.test.idx)


def test_bad_fractions_rejected(toy_xy):
    X, y = toy_xy
    with pytest.raises(ValueError):
        make_splits(X, y, {"train": 0.5, "calib_fit": 0.1, "calib": 0.2, "test": 0.3})
    with pytest.raises(ValueError):
        make_splits(X, y, {"train": 0.8, "calib": 0.1, "test": 0.1})
        