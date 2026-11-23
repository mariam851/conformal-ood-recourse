"""Actionability constraints and cost for tabular recourse.

A counterfactual may only change MUTABLE features. Immutable = dataset-level immutables (age, sex, ...),
extra immutables from the dataset config, and spike features (mostly one value, e.g. capital-gain; the same
rule used for the OOD scenarios, D7). Monotone-increasing features (education) may not decrease. Numeric
values stay inside the observed train range and integer-valued features stay integer.

Cost of a candidate x' for the original x:
    sum over changed numeric features |x'_f - x_f| / std_f   +   number of changed categorical features.
(The search adds a sparsity term on top.)

Populations are dicts {column: array}, which keeps the evolutionary search vectorised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from conformal_trees.data.ood_scenarios import _is_integer_valued, _mode_share


@dataclass
class RecourseSpace:
    columns: list[str]
    numeric: list[str]
    categorical: list[str]
    mutable: list[str]
    monotone_up: list[str]
    lower: dict[str, float]
    upper: dict[str, float]
    scale: dict[str, float]
    categories: dict[str, list]
    integer: list[str]
    spiky: list[str] = field(default_factory=list)

    @classmethod
    def from_dataset(
        cls,
        ds,
        X_train: pd.DataFrame,
        *,
        extra_immutable: Sequence[str] = (),
        monotone_up: Sequence[str] = (),
        spike_immutable: bool = True,
        max_mode_share: float = 0.75,
    ) -> "RecourseSpace":
        immutable = set(ds.immutable) | set(extra_immutable)
        spiky: list[str] = []
        if spike_immutable:
            spiky = [c for c in ds.numeric if _mode_share(X_train[c]) > max_mode_share]
            immutable |= set(spiky)
        unknown = (immutable | set(monotone_up)) - set(X_train.columns)
        if unknown:
            raise ValueError(f"Unknown features in recourse constraints: {sorted(unknown)}")
        std = X_train[ds.numeric].astype(float).std()
        return cls(
            columns=list(X_train.columns),
            numeric=list(ds.numeric),
            categorical=list(ds.categorical),
            mutable=[c for c in X_train.columns if c not in immutable],
            monotone_up=list(monotone_up),
            lower={c: float(X_train[c].min()) for c in ds.numeric},
            upper={c: float(X_train[c].max()) for c in ds.numeric},
            scale={c: float(std[c]) if np.isfinite(std[c]) and std[c] > 0 else 1.0 for c in ds.numeric},
            categories={c: list(pd.Categorical(X_train[c]).categories) for c in ds.categorical},
            integer=[c for c in ds.numeric if _is_integer_valued(X_train[c])],
            spiky=spiky,
        )

    # ---- conversions -------------------------------------------------------------------------------
    def row_to_dict(self, row: pd.Series) -> dict:
        return {c: (str(row[c]) if c in self.categorical else float(row[c])) for c in self.columns}

    def to_frame(self, pop: dict[str, np.ndarray]) -> pd.DataFrame:
        data = {}
        for c in self.columns:
            data[c] = (
                pd.Categorical(pop[c], categories=self.categories[c]) if c in self.categorical else pop[c]
            )
        return pd.DataFrame(data)

    # ---- constraints -------------------------------------------------------------------------------
    def project(self, pop: dict[str, np.ndarray], orig: dict) -> dict[str, np.ndarray]:
        """Enforce immutability, bounds, monotonicity and integrality (returns a new dict)."""
        out = {}
        n = len(next(iter(pop.values())))
        for c in self.columns:
            if c not in self.mutable:
                out[c] = np.full(n, orig[c], dtype=object if c in self.categorical else float)
            elif c in self.categorical:
                out[c] = pop[c]
            else:
                v = np.clip(pop[c].astype(float), self.lower[c], self.upper[c])
                if c in self.monotone_up:
                    v = np.maximum(v, orig[c])
                if c in self.integer:
                    v = np.round(v)
                out[c] = v
        return out

    # ---- cost --------------------------------------------------------------------------------------
    def changed_matrix(self, pop: dict[str, np.ndarray], orig: dict) -> np.ndarray:
        cols = [pop[c] != orig[c] for c in self.mutable]
        return np.column_stack(cols) if cols else np.zeros((len(next(iter(pop.values()))), 0), bool)

    def n_changed(self, pop, orig) -> np.ndarray:
        return self.changed_matrix(pop, orig).sum(axis=1)

    def cost(self, pop, orig) -> np.ndarray:
        n = len(next(iter(pop.values())))
        total = np.zeros(n)
        for c in self.mutable:
            if c in self.categorical:
                total += (pop[c] != orig[c]).astype(float)
            else:
                total += np.abs(pop[c].astype(float) - orig[c]) / self.scale[c]
        return total
    