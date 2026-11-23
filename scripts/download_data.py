"""Download raw datasets into data/raw/.

Usage (from the project root):
    python scripts/download_data.py --dataset adult
    python scripts/download_data.py --dataset mydata --openml-id 1234        # any OpenML dataset by id
    python scripts/download_data.py --dataset mydata --openml-name credit-g --version 1

The file is saved as data/raw/<dataset>.csv. Then run scripts/inspect_dataset.py on it to draft the config.
Check the columns before trusting a dataset: some OpenML versions have anonymised column names (V1, V2, ...),
which cannot be used for an actionability study.
"""
from __future__ import annotations

import argparse

from sklearn.datasets import fetch_openml

from conformal_trees.utils.config import load_config, resolve_path

# name -> OpenML data id. Only datasets whose schema has been checked belong here.
REGISTRY = {
    "adult": 1590,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="output name (data/raw/<dataset>.csv)")
    ap.add_argument("--openml-id", type=int, default=None)
    ap.add_argument("--openml-name", default=None)
    ap.add_argument("--version", default="active")
    args = ap.parse_args()

    if args.openml_id is not None:
        kwargs = {"data_id": args.openml_id}
    elif args.openml_name is not None:
        kwargs = {"name": args.openml_name, "version": args.version}
    elif args.dataset in REGISTRY:
        kwargs = {"data_id": REGISTRY[args.dataset]}
    else:
        raise SystemExit(f"'{args.dataset}' is not in the registry {sorted(REGISTRY)}; pass --openml-id or --openml-name.")

    raw_dir = resolve_path(load_config("base.yaml")["paths"]["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{args.dataset}.csv"

    print(f"Fetching from OpenML {kwargs} ...")
    df = fetch_openml(as_frame=True, parser="auto", **kwargs).frame
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} columns -> {out}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
    