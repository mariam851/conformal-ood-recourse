"""Plain OOD baselines (higher score = more OOD). They are the bar the new signals must clear."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from conformal_trees.data.encoding import TabularEncoder
from conformal_trees.utils.seed import derive_seed, make_rng


class IsolationForestScore:
    def __init__(self, numeric, categorical, n_estimators: int = 200, n_jobs: int = 4, seed: int = 0):
        self.encoder = TabularEncoder(numeric, categorical)
        self.model = IsolationForest(
            n_estimators=n_estimators, n_jobs=n_jobs, random_state=derive_seed(seed, "iforest")
        )

    def fit(self, X: pd.DataFrame) -> "IsolationForestScore":
        self.model.fit(self.encoder.fit(X).transform(X))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(self.encoder.transform(X))


def msp_score(p_pos: np.ndarray) -> np.ndarray:
    """1 - max softmax probability of the binary classifier (higher = less confident)."""
    p = np.asarray(p_pos, dtype=float)
    return 1.0 - np.maximum(p, 1.0 - p)


class LOFScore:
    """Local Outlier Factor (novelty mode) on standardised encoded features; higher score = more OOD.

    Used as an evaluator that is INDEPENDENT of the Isolation-Forest gate in the recourse study.
    """

    def __init__(self, numeric, categorical, n_neighbors: int = 20, n_jobs: int = 4):
        self.encoder = TabularEncoder(numeric, categorical)
        self.scaler = StandardScaler()
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=n_jobs)

    def fit(self, X: pd.DataFrame) -> "LOFScore":
        self.model.fit(self.scaler.fit_transform(self.encoder.fit(X).transform(X)))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(self.scaler.transform(self.encoder.transform(X)))


class KNNScore:
    """Mean distance to the k nearest training rows (standardised encoded features); higher = more OOD.

    A third, simple density-style detector, used to check whether a gate that targets one detector
    also helps detectors it was not built for.
    """

    def __init__(self, numeric, categorical, k: int = 10, n_jobs: int = 4):
        self.encoder = TabularEncoder(numeric, categorical)
        self.scaler = StandardScaler()
        self.nn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs)

    def fit(self, X: pd.DataFrame) -> "KNNScore":
        self.nn.fit(self.scaler.fit_transform(self.encoder.fit(X).transform(X)))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        dist, _ = self.nn.kneighbors(self.scaler.transform(self.encoder.transform(X)))
        return dist.mean(axis=1)


class GMMScore:
    """Negative log-likelihood under a Gaussian mixture on standardised encoded features; higher = more OOD.

    A density-based detector that is NOT used inside any gate: it is a held-out evaluator.
    """

    def __init__(self, numeric, categorical, n_components: int = 8, seed: int = 0):
        self.encoder = TabularEncoder(numeric, categorical)
        self.scaler = StandardScaler()
        self.model = GaussianMixture(
            n_components=n_components, covariance_type="full", reg_covar=1e-3,
            random_state=derive_seed(seed, "gmm"),
        )

    def fit(self, X: pd.DataFrame) -> "GMMScore":
        self.model.fit(self.scaler.fit_transform(self.encoder.fit(X).transform(X)))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(self.scaler.transform(self.encoder.transform(X)))


class OCSVMScore:
    """One-class SVM (RBF) on standardised encoded features, fit on a subsample; higher = more OOD.

    A boundary-based detector that is NOT used inside any gate: it is a held-out evaluator.
    """

    def __init__(self, numeric, categorical, nu: float = 0.05, max_fit_rows: int = 5000, seed: int = 0):
        self.encoder = TabularEncoder(numeric, categorical)
        self.scaler = StandardScaler()
        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
        self.max_fit_rows = max_fit_rows
        self.seed = seed

    def fit(self, X: pd.DataFrame) -> "OCSVMScore":
        Z = self.scaler.fit_transform(self.encoder.fit(X).transform(X))
        if len(Z) > self.max_fit_rows:
            idx = make_rng(derive_seed(self.seed, "ocsvm")).choice(len(Z), self.max_fit_rows, replace=False)
            Z = Z[idx]
        self.model.fit(Z)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(self.scaler.transform(self.encoder.transform(X)))
    