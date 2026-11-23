"""GBDT Wrapper for LightGBM configured for tabular benchmarks."""

from __future__ import annotations

from typing import Any
import lightgbm as lgb
import numpy as np
import pandas as pd
from packaging import version
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

_NEW_EVAL_API = version.parse(lgb.__version__) >= version.parse("4.0.0")


class GBDTClassifier(BaseEstimator, ClassifierMixin):
    """Clean LightGBM classifier wrapper supporting dict params, val_fraction, and early stopping."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        *,
        early_stopping_rounds: int | None = None,
        val_fraction: float = 0.0,
        seed: int = 0,
        **kwargs: Any,
    ):
        if not (0.0 <= val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

        self.params = dict(params or {})
        self.params.update(kwargs)

        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)

        if "lightgbm" in self.params and isinstance(self.params["lightgbm"], dict):
            self.params.update(self.params.pop("lightgbm"))

        cleaned = {}
        for k, v in self.params.items():
            if k in (
                "backend",
                "extra_trees",
                "extratrees",
                "val_fraction",
                "seed",
                "categorical_feature",
                "categorical_features",
                "cat_feature",
                "cat_features",
            ):
                continue
            if isinstance(v, dict):
                continue
            cleaned[k] = v

        cleaned.setdefault("verbose", -1)
        cleaned["random_state"] = self.seed
        self.lgb_params_ = cleaned

        self.model: lgb.LGBMClassifier | None = None
        self.classes_: np.ndarray | None = None
        self.best_iteration_: int | None = None
        self.feature_names_: list[str] = []

    @property
    def model_(self) -> lgb.LGBMClassifier | None:
        """Alias for sklearn-standard convention."""
        return self.model

    @property
    def booster_(self) -> lgb.Booster | None:
        """Direct access to the underlying LightGBM Booster object."""
        if self.model is None:
            return None
        return getattr(self.model, "booster_", None)

    @property
    def num_trees(self) -> int:
        """Return total number of trees/boosting iterations."""
        booster = self.booster_
        if booster is not None and hasattr(booster, "num_trees"):
            return booster.num_trees()
        if self.model is not None and hasattr(self.model, "n_estimators_"):
            return int(self.model.n_estimators_)
        return getattr(self.model, "n_estimators", 50) if self.model is not None else 0

    @property
    def num_iterations(self) -> int:
        """Alias for num_trees."""
        return self.num_trees

    @classmethod
    def from_config(cls, cfg: dict[str, Any], seed: int = 0) -> GBDTClassifier:
        c = dict(cfg)
        es = c.pop("early_stopping_rounds", None)
        vf = c.pop("val_fraction", 0.0)
        return cls(c, early_stopping_rounds=es, val_fraction=vf, seed=seed)

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        categorical_features: list[str] | None = None,
        eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> GBDTClassifier:
        self.classes_ = np.unique(y)
        X_fit = X.copy()
        self.feature_names_ = list(X_fit.columns)

        cat_cols = list(categorical_features) if categorical_features is not None else []
        if not cat_cols:
            cat_cols = list(X_fit.select_dtypes(include=["category", "object"]).columns)

        for c in cat_cols:
            if c in X_fit.columns:
                X_fit[c] = X_fit[c].astype("category")

        val_data = eval_set
        if val_data is None and self.val_fraction > 0.0 and self.early_stopping_rounds:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_fit, y, test_size=self.val_fraction, random_state=self.seed, stratify=y
            )
            X_fit, y = X_tr, y_tr
            val_data = (X_val, y_val)

        clean_params = {k: v for k, v in self.lgb_params_.items() if "categorical" not in k.lower()}
        self.model = lgb.LGBMClassifier(**clean_params)

        if val_data is not None and self.early_stopping_rounds:
            X_val, y_val = val_data
            X_val = X_val.copy()
            for c in cat_cols:
                if c in X_val.columns:
                    X_val[c] = X_val[c].astype("category")

            callbacks = [lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)]
            
            self.model.fit(
                X_fit,
                y,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
            self.best_iteration_ = getattr(self.model, "best_iteration_", None) or self.model.n_estimators
        else:
            self.model.fit(
                X_fit,
                y,
            )
            booster = getattr(self.model, "booster_", None)
            num_trees = booster.num_trees() if (booster is not None and hasattr(booster, "num_trees")) else 50
            self.best_iteration_ = getattr(self.model, "n_estimators", None) or num_trees

        return self

    def _validate_columns(self, X: pd.DataFrame) -> None:
        if list(X.columns) != self.feature_names_:
            raise ValueError(f"Columns differ from fit time. Expected {self.feature_names_}, got {list(X.columns)}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before predict.")
        self._validate_columns(X)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        self._validate_columns(X)
        return self.model.predict_proba(X)

    def get_leaf_indices(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before extracting leaves.")
        self._validate_columns(X)
        booster = self.model.booster_
        return booster.predict(X, pred_leaf=True)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model must be fitted before extracting feature importances.")
        booster = self.model.booster_
        importances = booster.feature_importance(importance_type=importance_type)
        return pd.Series(importances, index=self.feature_names_, name="importance").sort_values(ascending=False)


GBDTModel = GBDTClassifier
