"""Tabular Encoder: Handles feature scaling and categorical encoding.

Rules:
- Fits strictly on training split (no data leakage).
- Handles unseen categories gracefully (maps to Unknown).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class TabularEncoder(BaseEstimator, TransformerMixin):
    """Encodes tabular features (numeric scaling + category index mapping)."""

    def __init__(self, numeric_cols: list[str] | None = None, categorical_cols: list[str] | None = None):
        self.numeric_cols = list(numeric_cols) if numeric_cols is not None else []
        self.categorical_cols = list(categorical_cols) if categorical_cols is not None else []
        self.scaler = StandardScaler()
        self.category_maps_: dict[str, dict[str, int]] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> TabularEncoder:
        X_df = X.copy()
        
        # Fit scaler on numeric features
        if self.numeric_cols:
            self.scaler.fit(X_df[self.numeric_cols].fillna(0.0))

        # Build mapping for categoricals
        for col in self.categorical_cols:
            if col in X_df.columns:
                unique_vals = X_df[col].dropna().astype(str).unique()
                self.category_maps_[col] = {val: idx for idx, val in enumerate(unique_vals)}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # Transform numeric
        if self.numeric_cols:
            scaled = self.scaler.transform(X_out[self.numeric_cols].fillna(0.0))
            X_out[self.numeric_cols] = scaled

        # Transform categoricals (unseen becomes -1)
        for col, cat_map in self.category_maps_.items():
            if col in X_out.columns:
                s = X_out[col].astype(str)
                X_out[col] = s.map(cat_map).fillna(-1).astype(int)

        return X_out

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
    