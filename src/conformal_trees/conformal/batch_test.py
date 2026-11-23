"""Permutation test for "is this batch drawn from the in-distribution?".

Setting: a clean ID reference (the calibration rows) and an unlabeled batch of m rows. Under
H0 (the batch is iid from the same ID distribution as the reference) the pooled rows are exchangeable,
so any statistic computed on a random m-subset of the pool has the same distribution as the statistic
of the observed batch. With B random permutations,

    p_upper = (1 + #{perm >= obs}) / (B + 1),   p_lower = (1 + #{perm <= obs}) / (B + 1)

are EXACTLY valid p-values (P(p <= alpha) <= alpha, finite sample, any alpha), no matter how the statistic
depends on the rest of the batch. This is what the per-row conformal wrapper cannot offer for TOOD.

Scope of the guarantee: it is a test of the BATCH, not of individual rows. It needs the scoring models to
be fit on data independent of the pool (train), and the batch to be iid under H0.

Alternatives per statistic: "upper" (large = OOD) or "two" (p = min(1, 2 * min(p_upper, p_lower)), needs no
direction). TOOD's within-batch collision and the batch mean of per-row scores have no stable direction
(docs/decisions.md D10), so they use "two"; the MMD is non-negative in expectation under any difference.
"""
from __future__ import annotations

import numpy as np

from conformal_trees.signals.batch_stats import LeafIndex, leaf_counts, rf_mmd2, within_collision


def _pvalue(perm: np.ndarray, obs: float, alternative: str) -> float:
    b = len(perm)
    p_up = (1.0 + np.sum(perm >= obs)) / (b + 1.0)
    if alternative == "upper":
        return float(p_up)
    p_lo = (1.0 + np.sum(perm <= obs)) / (b + 1.0)
    return float(min(1.0, 2.0 * min(p_up, p_lo)))


def batch_permutation_test(
    global_ids: np.ndarray,
    m: int,
    leaf_index: LeafIndex,
    *,
    rng: np.random.Generator,
    n_perm: int = 199,
    scores: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """p-values of several batch statistics, computed on the SAME permutations.

    global_ids : leaf ids (LeafIndex.to_global) of the pool = reference rows first, then the m batch rows.
    scores     : optional per-row scores aligned with the pool; each adds a "mean_<name>" statistic.
    Returns {"tood": p, "rf_mmd": p, "mean_<name>": p, ...}.
    """
    n_pool = len(global_ids)
    n_ref = n_pool - m
    if m < 2 or n_ref < 2:
        raise ValueError("need at least 2 batch rows and 2 reference rows.")
    scores = scores or {}
    L, total = leaf_index.n_trees, leaf_index.total
    pool_counts = leaf_counts(global_ids, total)

    def stats(idx: np.ndarray) -> dict[str, float]:
        cb = leaf_counts(global_ids[idx], total)
        out = {
            "tood": within_collision(cb, m, L),
            "rf_mmd": rf_mmd2(cb, pool_counts - cb, m, n_ref, L),
        }
        for name, s in scores.items():
            out[f"mean_{name}"] = float(np.mean(s[idx]))
        return out

    alternative = {"tood": "two", "rf_mmd": "upper", **{f"mean_{k}": "two" for k in scores}}
    observed = stats(np.arange(n_ref, n_pool))
    perms = {k: np.empty(n_perm) for k in observed}
    for b in range(n_perm):
        s = stats(rng.permutation(n_pool)[:m])
        for k, v in s.items():
            perms[k][b] = v
    return {k: _pvalue(perms[k], observed[k], alternative[k]) for k in observed}
