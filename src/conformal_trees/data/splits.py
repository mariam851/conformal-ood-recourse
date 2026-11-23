"""Four-way stratified split: train / calib_fit / calib / test.

Why four and not three?
  * ``train``      : fits the GBDT and the ExtraTrees embedder.
  * ``calib_fit``  : held-out ID reference sample used by the OOD signals
                     (e.g. the reference set for APHD). Using training points
                     as the reference would bias the scores, since the trees
                     were fit on them.
  * ``calib``      : ID scores computed here become the conformal p-values.
  * ``test``       : final ID evaluation.
``calib_fit`` and ``calib`` must be disjoint, otherwise exchangeability between
calibration and test scores fails and the conformal guarantee is void.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from conformal_trees.utils.seed import derive_seed

SPLIT_NAMES = ("train", "calib_fit", "calib", "test")

DEFAULT_FRACTIONS = {"train": 0.5, "calib_fit": 0.1, "calib": 0.2, "test": 0.2}


@dataclass(frozen=True)
class Split:
    X: pd.DataFrame
    y: np.ndarray
    idx: np.ndarray        # positions of these rows in the original dataset

    def __len__(self) -> int:
        return len(self.y)


@dataclass(frozen=True)
class Splits:
    train: Split
    calib_fit: Split
    calib: Split
    test: Split

    def __iter__(self):
        for name in SPLIT_NAMES:
            yield name, getattr(self, name)

    def summary(self) -> pd.DataFrame:
        total = sum(len(s) for _, s in self)
        return pd.DataFrame(
            {
                "n": {name: len(s) for name, s in self},
                "fraction": {name: len(s) / total for name, s in self},
                "positive_rate": {name: float(s.y.mean()) for name, s in self},
            }
        )


def _validate_fractions(fractions: dict[str, float]) -> None:
    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(f"fractions must have exactly the keys {SPLIT_NAMES}, got {sorted(fractions)}")
    if any(f <= 0 for f in fractions.values()):
        raise ValueError(f"All split fractions must be > 0: {fractions}")
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {sum(fractions.values()):.4f}")


def make_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    fractions: dict[str, float] | None = None,
    seed: int = 0,
) -> Splits:
    """Stratified, reproducible, disjoint split. Same seed -> identical indices."""
    fractions = dict(DEFAULT_FRACTIONS if fractions is None else fractions)
    _validate_fractions(fractions)
    if len(X) != len(y):
        raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")
    y = np.asarray(y)

    remaining = np.arange(len(y))
    left = 1.0
    parts: dict[str, np.ndarray] = {}

    # Carve off test, then calib, then calib_fit; whatever is left is train.
    for name in ("test", "calib", "calib_fit"):
        remaining, part = train_test_split(
            remaining,
            test_size=fractions[name] / left,
            stratify=y[remaining],
            random_state=derive_seed(seed, "split", name),
        )
        parts[name] = np.sort(part)
        left -= fractions[name]
    parts["train"] = np.sort(remaining)

    _check_partition(parts, n=len(y))

    return Splits(
        **{
            name: Split(X=X.iloc[idx].reset_index(drop=True), y=y[idx], idx=idx)
            for name, idx in parts.items()
        }
    )


def _check_partition(parts: dict[str, np.ndarray], n: int) -> None:
    """Hard guarantee: splits are pairwise disjoint and cover every row exactly once."""
    all_idx = np.concatenate(list(parts.values()))
    if len(all_idx) != n or len(np.unique(all_idx)) != n:
        raise AssertionError("Splits overlap or do not cover the dataset; refusing to continue.")
    