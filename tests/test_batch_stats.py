import numpy as np
import pytest

from conformal_trees.conformal import batch_permutation_test
from conformal_trees.signals import (
    LeafIndex,
    cross_collision,
    leaf_counts,
    rf_mmd2,
    tood_batch_score,
    within_collision,
)


def _kernel(a, b):
    """RF kernel: fraction of trees in which two leaf vectors agree."""
    return float(np.mean(a == b))


def _setup(seed=0, m=30, n=45, n_trees=8, n_leaves=6):
    rng = np.random.default_rng(seed)
    batch = rng.integers(0, n_leaves, size=(m, n_trees))
    ref = rng.integers(0, n_leaves, size=(n, n_trees))
    li = LeafIndex([n_leaves] * n_trees)
    gb, gr = li.to_global(batch), li.to_global(ref)
    return batch, ref, li, leaf_counts(gb, li.total), leaf_counts(gr, li.total)


def test_global_leaf_ids_are_unique_across_trees():
    li = LeafIndex([5, 7, 3])
    g = li.to_global(np.array([[4, 6, 2], [0, 0, 0]]))
    assert g.tolist() == [[4, 5 + 6, 12 + 2], [0, 5, 12]]
    assert li.total == 15
    with pytest.raises(ValueError):
        li.to_global(np.zeros((2, 4), dtype=int))


def test_within_collision_matches_pairwise_definition_and_tood_scores():
    batch, _, li, cb, _ = _setup()
    m = len(batch)
    brute = np.mean([_kernel(batch[i], batch[j]) for i in range(m) for j in range(m) if i != j])
    assert within_collision(cb, m, li.n_trees) == pytest.approx(brute)
    assert within_collision(cb, m, li.n_trees) == pytest.approx(tood_batch_score(batch).mean())


def test_cross_collision_and_rf_mmd2_match_pairwise_definition():
    batch, ref, li, cb, cr = _setup(seed=1)
    m, n = len(batch), len(ref)
    cross = np.mean([_kernel(a, b) for a in batch for b in ref])
    assert cross_collision(cb, cr, m, n, li.n_trees) == pytest.approx(cross)
    w_r = np.mean([_kernel(ref[i], ref[j]) for i in range(n) for j in range(n) if i != j])
    expected = within_collision(cb, m, li.n_trees) + w_r - 2 * cross
    assert rf_mmd2(cb, cr, m, n, li.n_trees) == pytest.approx(expected)


def _pool(rng, n_ref, m, contamination=0.0, n_trees=10, n_leaves=40):
    """ID rows: leaves ~ skewed distribution; contaminated rows all land in the same leaves."""
    probs = 1.0 / np.arange(1, n_leaves + 1)
    probs /= probs.sum()
    ref = rng.choice(n_leaves, size=(n_ref, n_trees), p=probs)
    batch = rng.choice(n_leaves, size=(m, n_trees), p=probs)
    k = int(round(contamination * m))
    if k:
        batch[:k] = n_leaves - 1                                     # OOD cluster in a rare leaf
    li = LeafIndex([n_leaves] * n_trees)
    return li.to_global(np.vstack([ref, batch])), li


def test_permutation_test_is_valid_under_h0():
    rng = np.random.default_rng(0)
    reps, alpha = 150, 0.1
    rejected = {"tood": 0, "rf_mmd": 0}
    for _ in range(reps):
        g, li = _pool(rng, n_ref=150, m=60)
        p = batch_permutation_test(g, 60, li, rng=rng, n_perm=99)
        for k in rejected:
            rejected[k] += p[k] <= alpha
    for k, v in rejected.items():
        assert v / reps < alpha + 0.06, (k, v / reps)          # exact validity; slack for 150 replicates


def test_permutation_test_has_power_against_a_contaminated_batch():
    rng = np.random.default_rng(1)
    hits = 0
    reps = 40
    for _ in range(reps):
        g, li = _pool(rng, n_ref=150, m=60, contamination=0.4)
        hits += batch_permutation_test(g, 60, li, rng=rng, n_perm=99)["rf_mmd"] <= 0.05
    assert hits / reps > 0.8


def test_scores_add_mean_statistics_and_validation():
    rng = np.random.default_rng(2)
    g, li = _pool(rng, n_ref=100, m=40)
    scores = {"s": rng.normal(size=len(g))}
    p = batch_permutation_test(g, 40, li, rng=rng, n_perm=49, scores=scores)
    assert set(p) == {"tood", "rf_mmd", "mean_s"} and all(0 < v <= 1 for v in p.values())
    with pytest.raises(ValueError):
        batch_permutation_test(g, 1, li, rng=rng)
    with pytest.raises(ValueError):
        batch_permutation_test(g, len(g) - 1, li, rng=rng)
        