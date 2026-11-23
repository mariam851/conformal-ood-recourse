"""Inspect a raw CSV and draft configs/datasets/<name>.yaml for a new dataset.

Usage (from the project root):
    python scripts/inspect_dataset.py --file mydata.csv --target risk
    python scripts/inspect_dataset.py --file mydata.csv --target risk --name mydata > configs/datasets/mydata.yaml

--file is looked up in data/raw unless it is an absolute path. The YAML skeleton goes to stdout and the report to
stderr, so redirecting stdout gives a clean file. You MUST then edit it: positive_label (the favourable outcome),
drop_columns (identifiers, leakage, weights), immutable, and the `recourse:` block. Nothing here decides actionability for you.

It also prints how the conformal gate behaves at this dataset size: with the default 20% calibration split, the
calibration-conditional rule needs a few thousand calibration rows to be close to the plain alpha-level rule.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from conformal_trees.conformal import conditional_threshold, marginal_rank
from conformal_trees.utils import load_config, resolve_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cfg = load_config("base.yaml")
    path = Path(args.file)
    if not path.is_absolute():
        path = resolve_path(cfg["paths"]["data_raw"]) / path
    df = pd.read_csv(path, skipinitialspace=True).replace("?", np.nan)
    name = args.name or path.stem
    if args.target not in df.columns:
        raise SystemExit(f"target '{args.target}' not in columns: {list(df.columns)}")

    log = lambda *a: print(*a, file=sys.stderr)
    log(f"\n{path.name}: {df.shape[0]} rows x {df.shape[1]} columns")
    rows = []
    for c in df.columns:
        s = df[c]
        counts = s.dropna().value_counts(normalize=True)
        rows.append({"column": c, "dtype": str(s.dtype), "n_unique": s.nunique(dropna=True),
                     "missing": round(float(s.isna().mean()), 3),
                     "mode_share": round(float(counts.iloc[0]), 3) if len(counts) else np.nan})
    log(pd.DataFrame(rows).to_string(index=False))

    labels = df[args.target].astype(str).str.strip().str.rstrip(".")
    log(f"\ntarget '{args.target}' value counts:\n{labels.value_counts().to_string()}")

    feats = [c for c in df.columns if c != args.target]
    numeric = [c for c in feats if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    categorical = [c for c in feats if c not in numeric]
    review = [c for c in numeric if df[c].nunique() <= 6]
    spiky = [c for c in numeric if len(df[c].dropna()) and df[c].dropna().value_counts(normalize=True).iloc[0] > 0.75]
    wide = [c for c in categorical if df[c].nunique() > 50]
    ids = [c for c in feats if df[c].nunique() == len(df)]
    log("\nREVIEW before using the skeleton:")
    log(f"  low-cardinality numeric (numeric or categorical?): {review}")
    log(f"  spike-at-a-value numeric (will be non-actionable): {spiky}")
    log(f"  very wide categorical (identifier? drop it?)     : {wide}")
    log(f"  one distinct value per row (identifier? drop it?): {ids}")

    n_cal = int(round(0.2 * len(df)))
    k_c, thr = conditional_threshold(n_cal, cfg["conformal"]["alpha"], cfg["conformal"]["delta"])
    k_m = marginal_rank(n_cal, cfg["conformal"]["alpha"])
    log(f"\nGate sanity at n_cal ~ {n_cal}: conditional rank k = {k_c} (plain alpha-test rank {k_m}); "
        f"effective FPR level {thr:.4f} vs alpha {cfg['conformal']['alpha']}")
    if k_c < 0.8 * k_m:
        log("  -> the calibration set is small: the conditional gate is noticeably stricter than alpha. Prefer a larger dataset.")

    log("\nYAML skeleton on stdout. Fill every FILL.")
    q = lambda xs: "[" + ", ".join(xs) + "]"
    print(f"name: {name}")
    print(f"raw_file: {path.name}")
    print(f"target: {args.target}")
    print(f'positive_label: "FILL"        # the FAVOURABLE outcome. Observed labels: {sorted(labels.unique())[:10]}')
    print("drop_columns: []              # FILL: identifiers, leakage, sampling weights")
    print(f"numeric: {q(numeric)}")
    print(f"categorical: {q(categorical)}")
    print("immutable: []                 # FILL: features the individual cannot change (age, sex, race, nationality, ...)")
    print("\nrecourse:")
    print("  extra_immutable: []         # FILL: changeable in principle but not actionable advice")
    print("  monotone_increasing: []     # FILL: features that may only increase (education, ...)")


if __name__ == "__main__":
    main()
    