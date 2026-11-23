"""Scenario diagnostics: how separable are an ID set and its shifted copy?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from conformal_trees.utils.seed import derive_seed


def shift_detectability(
    X_id: pd.DataFrame,
    X_ood: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_estimators: int = 100,
    seed: int = 0,
) -> float:
    """Classifier two-sample test (C2ST) AUROC between a paired ID set and OOD set.

    ~0.5 -> the two sets are indistinguishable; ~1.0 -> trivially separable.
    It is a *supervised* reference (the classifier sees OOD labels), so unsupervised
    detectors should not be expected to beat it by much.

    Rows are grouped by their original row: a row and its shifted copy always fall
    on the same side of every fold, otherwise the classifier could memorise pairs.
    """
    import lightgbm as lgb

    if len(X_id) != len(X_ood):
        raise ValueError("X_id and X_ood must be paired (same length).")
    n = len(X_id)
    X = pd.concat([X_id, X_ood], ignore_index=True)
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    groups = np.concatenate([np.arange(n), np.arange(n)])

    oof = np.zeros(2 * n)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            num_leaves=15,
            random_state=derive_seed(seed, "c2st"),
            deterministic=True,
            force_row_wise=True,
            n_jobs=4,
            verbose=-1,
        )
        clf.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]
    return float(roc_auc_score(y, oof))
