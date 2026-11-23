"""Numeric encoding for models that need a numeric matrix (ExtraTrees, IsolationForest).

Fit on ``train`` only (see docs/decisions.md D4). Numeric NaN -> train median.
Categoricals -> ordinal codes by category label; labels unseen at fit time -> -1.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


class TabularEncoder:
    def __init__(self, numeric: Sequence[str], categorical: Sequence[str]) -> None:
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.medians_: dict[str, float] | None = None
        self.categories_: dict[str, list] | None = None

    def fit(self, X: pd.DataFrame) -> "TabularEncoder":
        self.medians_ = {c: float(X[c].median()) for c in self.numeric}
        self.categories_ = {c: list(pd.Categorical(X[c]).categories) for c in self.categorical}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None or self.categories_ is None:
            raise RuntimeError("Call fit() first.")
        cols = []
        for c in self.numeric:
            cols.append(X[c].astype(float).fillna(self.medians_[c]).to_numpy())
        for c in self.categorical:
            mapping = {cat: i for i, cat in enumerate(self.categories_[c])}
            codes = X[c].astype(object).map(mapping).fillna(-1.0)    # unseen / missing -> -1
            cols.append(codes.to_numpy(dtype=float))
        return np.column_stack(cols)

    @property
    def feature_names_(self) -> list[str]:
        return self.numeric + self.categorical
    