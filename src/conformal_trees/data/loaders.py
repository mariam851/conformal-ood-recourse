"""Load a raw tabular dataset into a clean, typed ``TabularDataset``.

Cleaning is deliberately minimal and config-driven (see configs/datasets/*.yaml).
Encoding / scaling is NOT done here: it must be fit on the training split only,
so it lives downstream of ``make_splits``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from conformal_trees.utils.config import (
    load_config,
    load_dataset_config,
    resolve_path,
)

UNKNOWN = "Unknown"


@dataclass(repr=False)
class TabularDataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray                       # int {0, 1}; 1 = favourable outcome
    numeric: list[str]
    categorical: list[str]
    immutable: list[str] = field(default_factory=list)
    positive_label: str = ""

    @property
    def mutable(self) -> list[str]:
        return [c for c in self.X.columns if c not in self.immutable]

    def __len__(self) -> int:
        return len(self.y)

    def __repr__(self) -> str:
        return (
            f"TabularDataset(name={self.name!r}, n={len(self)}, "
            f"features={self.X.shape[1]} ({len(self.numeric)} numeric, "
            f"{len(self.categorical)} categorical), positive_rate={self.y.mean():.3f})"
        )


def _validate_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    declared = set(cfg["numeric"]) | set(cfg["categorical"])
    present = set(df.columns)

    missing = declared - present
    if missing:
        raise ValueError(f"Columns declared in config but absent from the data: {sorted(missing)}")

    undeclared = present - declared
    if undeclared:
        raise ValueError(
            f"Columns in the data that are neither declared nor dropped: {sorted(undeclared)}. "
            "Add them to 'numeric', 'categorical' or 'drop_columns' in the dataset config."
        )

    bad_immutable = set(cfg.get("immutable", [])) - declared
    if bad_immutable:
        raise ValueError(f"'immutable' lists unknown features: {sorted(bad_immutable)}")


def load_dataset(
    name: str,
    *,
    dataset_cfg: dict[str, Any] | None = None,
    raw_dir: str | Path | None = None,
) -> TabularDataset:
    """Load ``data/raw/<raw_file>`` according to ``configs/datasets/<name>.yaml``.

    ``dataset_cfg`` and ``raw_dir`` can be passed explicitly (used by tests).
    """
    cfg = dataset_cfg if dataset_cfg is not None else load_dataset_config(name)
    if raw_dir is None:
        raw_dir = resolve_path(load_config("base.yaml")["paths"]["data_raw"])
    path = Path(raw_dir) / cfg["raw_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py --dataset {name}` first."
        )

    df = pd.read_csv(path, skipinitialspace=True)
    df = df.replace("?", np.nan)  # UCI-style missing marker

    target = cfg["target"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in {path.name}: {list(df.columns)}")

    df = df.dropna(subset=[target]).reset_index(drop=True)

    # Some UCI files end labels with a trailing dot (">50K.").
    labels = df[target].astype(str).str.strip().str.rstrip(".")
    y = (labels == cfg["positive_label"]).to_numpy().astype(int)
    if y.min() == y.max():
        raise ValueError(
            f"Target is constant after mapping positive_label={cfg['positive_label']!r}; "
            f"observed labels: {sorted(labels.unique())[:10]}"
        )

    X = df.drop(columns=[target, *cfg.get("drop_columns", [])])
    _validate_columns(X, cfg)

    for col in cfg["numeric"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in cfg["categorical"]:
        X[col] = X[col].astype("string").str.strip().fillna(UNKNOWN).astype("category")

    X = X[cfg["numeric"] + cfg["categorical"]]  # fixed, config-defined column order

    return TabularDataset(
        name=cfg["name"],
        X=X,
        y=y,
        numeric=list(cfg["numeric"]),
        categorical=list(cfg["categorical"]),
        immutable=list(cfg.get("immutable", [])),
        positive_label=cfg["positive_label"],
    )
