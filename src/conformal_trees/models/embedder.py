"""ExtraTrees leaf embedder (the tree embedding used by TOOD).

A sample x is mapped to T(x) = [leaf reached in tree 1, ..., leaf reached in tree L].
The trees are fit on in-distribution ``train`` rows only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from conformal_trees.data.encoding import TabularEncoder
from conformal_trees.utils.seed import derive_seed


class TreeEmbedder:
    def __init__(
        self,
        numeric: list[str],
        categorical: list[str],
        n_estimators: int = 100,
        min_samples_leaf: int = 1,
        max_features: str | float | int = "sqrt",
        n_jobs: int = 4,
        seed: int = 0,
    ) -> None:
        self.encoder = TabularEncoder(numeric, categorical)
        self.params = dict(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
        )
        self.seed = seed
        self.model_: ExtraTreesClassifier | None = None

    @classmethod
    def from_config(
        cls, emb_cfg: dict[str, Any], numeric: list[str], categorical: list[str], seed: int = 0
    ) -> "TreeEmbedder":
        return cls(numeric, categorical, seed=seed, **emb_cfg)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TreeEmbedder":
        Xe = self.encoder.fit(X).transform(X)
        self.model_ = ExtraTreesClassifier(
            **self.params, random_state=derive_seed(self.seed, "embedder")
        ).fit(Xe, np.asarray(y))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Leaf-index matrix of shape (n_samples, n_trees)."""
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        return self.model_.apply(self.encoder.transform(X))
    