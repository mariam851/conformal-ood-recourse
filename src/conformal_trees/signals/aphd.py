"""Average Pairwise Hamming Distance (APHD) on tree embeddings, in leaf-mass form."""
from __future__ import annotations

import numpy as np


def leaf_mass(
    query: np.ndarray, reference: np.ndarray, *, query_is_reference: bool = False
) -> np.ndarray:
    """Mean over trees of the fraction of reference rows sharing the query's leaf.

    query, reference: leaf-index matrices (n, L). If ``query_is_reference`` the query rows
    ARE the reference rows and each row is scored leave-one-out (self excluded).
    """
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference must have the same number of trees.")
    n_trees, n_ref = reference.shape[1], reference.shape[0]
    if query_is_reference and query.shape[0] != n_ref:
        raise ValueError("query_is_reference requires query is reference.")

    total = np.zeros(len(query), dtype=float)
    for t in range(n_trees):
        size = int(max(reference[:, t].max(), query[:, t].max())) + 1
        counts = np.bincount(reference[:, t], minlength=size)
        total += counts[query[:, t]]

    if query_is_reference:
        if n_ref < 2:
            raise ValueError("Leave-one-out scoring needs at least 2 rows.")
        return (total - n_trees) / (n_trees * (n_ref - 1))
    return total / (n_trees * n_ref)


def aphd(query: np.ndarray, reference: np.ndarray, *, query_is_reference: bool = False) -> np.ndarray:
    """Average pairwise Hamming distance to the reference set (= 1 - leaf mass)."""
    return 1.0 - leaf_mass(query, reference, query_is_reference=query_is_reference)


def tood_batch_score(batch: np.ndarray) -> np.ndarray:
    """Original TOOD (transductive): leaf mass of each row against the rest of its own batch."""
    return leaf_mass(batch, batch, query_is_reference=True)


def reference_score(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Inductive variant: APHD to a fixed ID reference set (use calib_fit). Higher = more OOD."""
    return aphd(query, reference)
