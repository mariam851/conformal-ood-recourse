"""Batch-level statistics on tree embeddings (random-forest kernel).

Kernel: k(x, x') = fraction of trees in which x and x' fall in the same leaf  (= 1 - Hamming distance).
All statistics below are functions of per-leaf COUNTS, so they cost O(n_trees * n_rows) instead of
O(n_rows^2 * n_trees). To count all trees in one pass, leaf ids are made globally unique
(``LeafIndex``): tree t, leaf l  ->  offset_t + l.

  within_collision(B)  = mean of k over pairs of DISTINCT rows of B
                         = mean of the TOOD leaf-mass scores of the batch (tests/test_batch_stats.py).
  cross_collision(A,B) = mean of k over pairs (a, b).
  rf_mmd2(B, R)        = within(B) + within(R) - 2 * cross(B, R)   (unbiased MMD^2 with the RF kernel).

within_collision(B) is TOOD's own quantity: it is large when the batch has rows clustered in shared
leaves, and it ignores the ID reference. rf_mmd2 also uses the reference, so it also reacts to a batch that
lands in leaves the reference rarely visits.

Honest note on novelty: a kernel two-sample statistic with a forest kernel is a standard construction in
two-sample testing. What this module adds is the cheap count-based form and its use inside a valid
permutation test for OOD batches; it must be compared with classifier two-sample tests before any claim.
"""
from __future__ import annotations

import numpy as np


class LeafIndex:
    """Globally unique leaf ids across the trees of a fitted ExtraTrees embedder."""

    def __init__(self, node_counts) -> None:
        node_counts = np.asarray(node_counts, dtype=np.int64)
        self.n_trees = len(node_counts)
        self.total = int(node_counts.sum())
        self.offsets = np.concatenate([[0], np.cumsum(node_counts)[:-1]])

    @classmethod
    def from_embedder(cls, embedder) -> "LeafIndex":
        return cls([t.tree_.node_count for t in embedder.model_.estimators_])

    def to_global(self, leaves: np.ndarray) -> np.ndarray:
        if leaves.shape[1] != self.n_trees:
            raise ValueError("leaf matrix has a different number of trees.")
        return leaves.astype(np.int64) + self.offsets[None, :]


def leaf_counts(global_ids: np.ndarray, total: int) -> np.ndarray:
    """Number of rows in every (tree, leaf) cell, as floats."""
    return np.bincount(global_ids.ravel(), minlength=total).astype(float)


def within_collision(counts: np.ndarray, m: int, n_trees: int) -> float:
    if m < 2:
        raise ValueError("need at least 2 rows.")
    return float((counts * (counts - 1.0)).sum() / (m * (m - 1.0)) / n_trees)


def cross_collision(counts_a: np.ndarray, counts_b: np.ndarray, m_a: int, m_b: int, n_trees: int) -> float:
    return float((counts_a * counts_b).sum() / (float(m_a) * m_b) / n_trees)


def rf_mmd2(counts_b: np.ndarray, counts_r: np.ndarray, m: int, n: int, n_trees: int) -> float:
    """Unbiased MMD^2 between a batch (m rows) and a reference (n rows) with the RF kernel."""
    return (
        within_collision(counts_b, m, n_trees)
        + within_collision(counts_r, n, n_trees)
        - 2.0 * cross_collision(counts_b, counts_r, m, n, n_trees)
    )
