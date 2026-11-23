"""Model-agnostic evolutionary search for tabular counterfactuals (recourse).

Goal: for a row x with P(y=1) < target, find x' that is actionable (constraints.py), reaches
P(y=1 | x') >= target, is cheap (cost + sparsity), and, optionally, passes an OOD gate (gate.py).

Fitness (lower is better), per candidate:
    cost + sparsity_weight * n_changed
        + penalty * max(0, target - P(y=1))          # not valid yet
        + penalty * gate.violation                   # flagged by the OOD gate
The best FEASIBLE candidate seen in any generation is returned (valid, gate passed, at least one change),
so the penalties only guide the search; they never decide feasibility.

Variation: tournament selection, uniform crossover, and mutation that either copies the feature from a real
favourable training row (a "donor", which keeps values data-like) or perturbs it (numeric: Gaussian step,
categorical: random level). It needs only ``model.predict_proba``, so it works for any classifier.
Same ``rng`` seed -> identical result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from conformal_trees.recourse.constraints import RecourseSpace
from conformal_trees.recourse.gate import OODGate


@dataclass(frozen=True)
class SearchConfig:
    target_prob: float = 0.5
    pop_size: int = 150
    generations: int = 30
    elite_frac: float = 0.1
    tournament: int = 3
    mutation_rate: float = 0.3
    sparsity_weight: float = 0.1
    penalty: float = 100.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SearchConfig":
        return cls(target_prob=cfg["recourse"]["target_prob"], **cfg["search"])


@dataclass
class RecourseResult:
    found: bool
    x_cf: dict | None
    prob: float
    cost: float
    n_changed: int
    violation: float
    n_evals: int


def make_donors(space: RecourseSpace, X_train: pd.DataFrame, y_train: np.ndarray) -> dict[str, np.ndarray]:
    """Columns of the favourable (y=1) training rows, as arrays."""
    pos = X_train[np.asarray(y_train) == 1]
    return {
        c: (pos[c].astype(object).to_numpy() if c in space.categorical else pos[c].astype(float).to_numpy())
        for c in space.columns
    }


def _mutate(pop, space, donors, rate, rng, n):
    for c in space.mutable:
        mask = rng.random(n) < rate
        k = int(mask.sum())
        if not k:
            continue
        use_donor = rng.random(k) < 0.5
        new = pop[c][mask].copy()
        donor_vals = donors[c][rng.integers(0, len(donors[c]), size=k)]
        if c in space.categorical:
            random_levels = np.array(space.categories[c], dtype=object)[
                rng.integers(0, len(space.categories[c]), size=k)
            ]
            new = np.where(use_donor, donor_vals, random_levels)
        else:
            step = new.astype(float) + rng.normal(0.0, 0.5 * space.scale[c], size=k)
            new = np.where(use_donor, donor_vals.astype(float), step)
        pop[c] = pop[c].copy()
        pop[c][mask] = new
    return pop


def find_recourse(
    model,
    space: RecourseSpace,
    x_row: pd.Series,
    donors: dict[str, np.ndarray],
    *,
    cfg: SearchConfig,
    rng: np.random.Generator,
    gate: OODGate | None = None,
) -> RecourseResult:
    orig = space.row_to_dict(x_row)
    if not space.mutable:
        return RecourseResult(False, None, float("nan"), float("inf"), 0, float("nan"), 0)

    P = cfg.pop_size
    pop = {c: np.full(P, orig[c], dtype=object if c in space.categorical else float) for c in space.columns}
    pop = _mutate(pop, space, donors, 0.5, rng, P)
    n_elite = max(1, int(round(cfg.elite_frac * P)))

    best = None            # (value, individual dict, prob, cost, n_changed, violation)
    n_evals = 0
    for _ in range(cfg.generations):
        pop = space.project(pop, orig)
        frame = space.to_frame(pop)
        prob = model.predict_proba(frame)[:, 1]
        cost = space.cost(pop, orig)
        nch = space.n_changed(pop, orig)
        viol = gate.violation(frame) if gate is not None else np.zeros(P)
        n_evals += P

        base = cost + cfg.sparsity_weight * nch
        fit = base + cfg.penalty * np.maximum(0.0, cfg.target_prob - prob) + cfg.penalty * viol
        feasible = (prob >= cfg.target_prob) & (viol == 0.0) & (nch > 0)
        if feasible.any():
            i = int(np.flatnonzero(feasible)[np.argmin(base[feasible])])
            if best is None or base[i] < best[0]:
                best = (float(base[i]), {c: pop[c][i] for c in space.columns}, float(prob[i]),
                        float(cost[i]), int(nch[i]), float(viol[i]))

        order = np.argsort(fit)
        n_off = P - n_elite
        cand = rng.integers(0, P, size=(n_off, 2, cfg.tournament))
        pa = cand[:, 0][np.arange(n_off), np.argmin(fit[cand[:, 0]], axis=1)]
        pb = cand[:, 1][np.arange(n_off), np.argmin(fit[cand[:, 1]], axis=1)]
        child = {}
        for c in space.columns:
            mask = rng.random(n_off) < 0.5
            child[c] = np.where(mask, pop[c][pa], pop[c][pb])
        child = _mutate(child, space, donors, cfg.mutation_rate, rng, n_off)
        pop = {c: np.concatenate([pop[c][order[:n_elite]], child[c]]) for c in space.columns}

    if best is None:
        return RecourseResult(False, None, float("nan"), float("inf"), 0, float("nan"), n_evals)
    _, x_cf, prob, cost, nch, viol = best
    return RecourseResult(True, x_cf, prob, cost, nch, viol, n_evals)
