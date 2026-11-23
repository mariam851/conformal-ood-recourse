"""Build every table (CSV + LaTeX) and figure (PNG + PDF) for the paper from the benchmark CSVs.

Usage (from the project root):
    python scripts/make_paper_artifacts.py
    python scripts/make_paper_artifacts.py --bench results/raw/recourse_benchmark.csv \
        --gate-bench results/raw/gate_benchmark.csv --target 0.6

Tables go to results/tables, figures to paper/figures. Whatever input file is missing is skipped,
so it is safe to run after each experiment. Nothing here changes any number: it only reads the CSVs.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

from conformal_trees.evaluation.paper_artifacts import make_all  # noqa: E402
from conformal_trees.utils import resolve_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default="results/raw/recourse_benchmark.csv")
    ap.add_argument("--gate-bench", default="results/raw/gate_benchmark.csv")
    ap.add_argument("--tables-dir", default="results/tables")
    ap.add_argument("--figures-dir", default="paper/figures")
    ap.add_argument("--target", type=float, default=0.6, help="headline search target for the per-target figures")
    args = ap.parse_args()

    made = make_all(
        bench_csv=resolve_path(args.bench), gate_csv=resolve_path(args.gate_bench),
        tables_dir=resolve_path(args.tables_dir), figures_dir=resolve_path(args.figures_dir),
        headline_target=args.target,
    )
    if not made:
        raise SystemExit("No input CSV found. Run scripts/run_recourse_benchmark.py or scripts/run_gate_benchmark.py first.")
    for name, files in made.items():
        print(f"{name:34s}", ", ".join(f.name for f in files))


if __name__ == "__main__":
    main()
    