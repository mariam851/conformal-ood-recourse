"""Gate benchmark: dataset-agnostic, multi-seed, resumable. One CSV row per (seed, variant, target, gate, individual).

Compares the same evolutionary recourse search with no gate and with OOD gates built on different detectors:
    gate_IF, gate_LOF, gate_kNN  : one detector each (conditional conformal rule, calibrated on `calib`)
    gate_ALL                     : conjunction of IF, LOF and kNN (a counterfactual must pass all three)
Every counterfactual is scored by ALL detectors (plain alpha-level conformal test). With --heldout, two more
detectors that are never used inside any gate (GMM density, one-class SVM) are evaluated too: they answer
"does a gate generalise to a detector it was not built on?".

Examples (from the project root):
    # held-out experiment on Adult, the recommended margin only
    python scripts/run_gate_benchmark.py --seeds 0 1 2 3 4 --n 200 --targets 0.6 --heldout --out results/raw/gate_benchmark.csv
    # ablations: tag every run, all go to the same file (the resume key is (seed, tag))
    python scripts/run_gate_benchmark.py --seeds 0 1 2 --n 100 --targets 0.6 --gates none gate_IF --tag gen10  --generations 10  --out results/raw/ablations.csv
    python scripts/run_gate_benchmark.py --seeds 0 1 2 --n 100 --targets 0.6 --gates none gate_IF --tag gen100 --generations 100 --out results/raw/ablations.csv
    python scripts/run_gate_benchmark.py --seeds 0 1 2 --n 100 --targets 0.6 --gates none gate_IF --tag relax --relax marital-status relationship --out results/raw/ablations.csv
    # another dataset (needs configs/datasets/<name>.yaml with a `recourse` block, see scripts/inspect_dataset.py)
    python scripts/run_gate_benchmark.py --dataset mydata --seeds 0 1 2 --n 200 --targets 0.6 --out results/raw/gate_benchmark_mydata.csv

Outputs: <out> and <out stem>_reference.csv (detector flag rates of the original and the real favourable rows).
Never mix runs with different columns (for example with and without --heldout) in one file: the script refuses.
"""
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import replace

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
warnings.filterwarnings("ignore", message=".*'eval_set' is deprecated.*")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from conformal_trees.conformal import conformal_pvalues  # noqa: E402
from conformal_trees.data import load_dataset, make_splits  # noqa: E402
from conformal_trees.evaluation import robust_validity  # noqa: E402
from conformal_trees.models import GBDTClassifier  # noqa: E402
from conformal_trees.recourse import (  # noqa: E402
    AllOfGate, OODGate, RecourseSpace, SearchConfig, find_recourse, make_donors,
)
from conformal_trees.signals import (  # noqa: E402
    GMMScore, IsolationForestScore, KNNScore, LOFScore, OCSVMScore,
)
from conformal_trees.utils import (  # noqa: E402
    derive_seed, load_config, load_dataset_config, make_rng, resolve_path,
)

GATES = ["none", "gate_IF", "gate_LOF", "gate_kNN", "gate_ALL"]


def expected_columns(heldout: bool) -> tuple[list[str], list[str]]:
    flags = ["flag_IF", "flag_LOF", "flag_kNN"] + (["flag_GMM", "flag_OCSVM"] if heldout else [])
    main = ["dataset", "seed", "variant", "target", "gate", "i", "idx", "found", "prob", "cost", "n_changed",
            *flags, "robust"]
    ref = ["dataset", "seed", "variant", "group", "n", *flags]
    return main, ref


def run_seed(seed: int, args, cfg, ds, rc):
    alpha, delta = cfg["gate"]["alpha"], cfg["gate"]["delta"]
    sp = make_splits(ds.X, ds.y, cfg["splits"], seed=seed)
    model = GBDTClassifier.from_config(cfg["gbdt"], seed=seed).fit(sp.train.X, sp.train.y)
    alt = [GBDTClassifier.from_config(cfg["gbdt"], seed=derive_seed(seed, "alt", i)).fit(sp.train.X, sp.train.y)
           for i in range(args.n_alt)]

    # n_jobs=1: the search scores ~150 rows at a time, where thread start-up costs more than it saves
    dets = {
        "IF": IsolationForestScore(ds.numeric, ds.categorical, n_jobs=1, seed=seed).fit(sp.train.X),
        "LOF": LOFScore(ds.numeric, ds.categorical, n_jobs=1).fit(sp.train.X),
        "kNN": KNNScore(ds.numeric, ds.categorical, n_jobs=1).fit(sp.train.X),
    }
    if args.heldout:
        dets["GMM"] = GMMScore(ds.numeric, ds.categorical, seed=seed).fit(sp.train.X)
        dets["OCSVM"] = OCSVMScore(ds.numeric, ds.categorical, seed=seed).fit(sp.train.X)
    cal = {k: d.score(sp.calib.X) for k, d in dets.items()}

    single = {f"gate_{k}": OODGate(dets[k].score, cal[k], alpha, delta) for k in ("IF", "LOF", "kNN")}
    all_gates = {"none": None, **single, "gate_ALL": AllOfGate(list(single.values()))}
    gates = {k: v for k, v in all_gates.items() if k in args.gates}

    extra_immutable = [f for f in rc.get("extra_immutable", []) if f not in args.relax]
    space = RecourseSpace.from_dataset(
        ds, sp.train.X, extra_immutable=extra_immutable, monotone_up=rc.get("monotone_increasing", []),
        spike_immutable=cfg["recourse"]["spike_immutable"], max_mode_share=cfg["recourse"]["max_mode_share"],
    )
    donors = make_donors(space, sp.train.X, sp.train.y)
    search_cfg = SearchConfig.from_config(cfg)
    if args.generations:
        search_cfg = replace(search_cfg, generations=args.generations)
    if args.pop_size:
        search_cfg = replace(search_cfg, pop_size=args.pop_size)

    p_test = model.predict_proba(sp.test.X)[:, 1]
    neg = np.flatnonzero(p_test < 0.5)
    picked = make_rng(derive_seed(seed, "bench", "pick")).choice(neg, size=min(args.n, len(neg)), replace=False)

    def flags(F):
        return {k: (conformal_pvalues(cal[k], d.score(F)) <= alpha).astype(float) for k, d in dets.items()}

    rows = []
    for t in args.targets:
        cfg_t = replace(search_cfg, target_prob=t)
        for gname, g in gates.items():
            res = [find_recourse(model, space, sp.test.X.iloc[idx], donors, cfg=cfg_t,
                                 rng=make_rng(derive_seed(seed, "bench", int(round(t * 100)), i)), gate=g)
                   for i, idx in enumerate(picked)]
            found_i = [i for i, r in enumerate(res) if r.found]
            extra = {}
            if found_i:
                F = space.to_frame({c: np.array([res[i].x_cf[c] for i in found_i],
                                                dtype=object if c in space.categorical else float)
                                    for c in space.columns})
                fl, rob = flags(F), robust_validity(alt, F, 0.5)
                extra = {i: {**{f"flag_{k}": float(fl[k][j]) for k in dets}, "robust": float(rob[j])}
                         for j, i in enumerate(found_i)}
            for i, (idx, r) in enumerate(zip(picked, res)):
                rows.append({"dataset": ds.name, "seed": seed, "variant": args.tag, "target": t, "gate": gname,
                             "i": i, "idx": int(idx), "found": r.found, "prob": r.prob,
                             "cost": r.cost if r.found else np.nan,
                             "n_changed": r.n_changed if r.found else np.nan, **extra.get(i, {})})

    real_pos = sp.test.X[p_test >= 0.5]
    ref = []
    for label, Xr in (("original_unfavourable", sp.test.X.iloc[picked]), ("real_favourable", real_pos)):
        fl = flags(Xr)
        ref.append({"dataset": ds.name, "seed": seed, "variant": args.tag, "group": label, "n": len(Xr),
                    **{f"flag_{k}": float(v.mean()) for k, v in fl.items()}})
    info = f"n_cal={len(sp.calib)}, gate k (IF)={single['gate_IF'].k}, unfavourable test rows={len(neg)}"
    return pd.DataFrame(rows), pd.DataFrame(ref), info


def _check_header(path, expected: list[str]) -> None:
    if path.exists():
        header = list(pd.read_csv(path, nrows=0).columns)
        if header != expected:
            raise SystemExit(
                f"{path} has different columns than this run (for example --heldout was used in only one of them).\n"
                f"  file: {header}\n  run : {expected}\nUse another --out file."
            )


def _append(df: pd.DataFrame, path, expected: list[str]) -> None:
    df = df.reindex(columns=expected)          # stable header even if a seed found no counterfactual at all
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="adult")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n", type=int, default=200, help="individuals per seed")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.6])
    ap.add_argument("--gates", nargs="+", default=["none", "gate_IF", "gate_LOF", "gate_ALL"], choices=GATES)
    ap.add_argument("--heldout", action="store_true", help="also evaluate the GMM and one-class SVM detectors")
    ap.add_argument("--tag", default="main", help="variant label stored in the CSV (used for ablations)")
    ap.add_argument("--generations", type=int, default=0, help="override the search generations")
    ap.add_argument("--pop-size", type=int, default=0, help="override the search population size")
    ap.add_argument("--relax", nargs="*", default=[], help="features removed from extra_immutable")
    ap.add_argument("--n-alt", type=int, default=3, help="re-trained models used for the robustness check")
    ap.add_argument("--out", default="results/raw/gate_benchmark.csv")
    args = ap.parse_args()

    cfg = load_config("base.yaml", "models.yaml", "recourse.yaml")
    ds = load_dataset(args.dataset)
    rc = load_dataset_config(args.dataset).get("recourse")
    if rc is None:
        raise SystemExit(f"configs/datasets/{args.dataset}.yaml needs a `recourse:` block "
                         "(extra_immutable, monotone_increasing). See scripts/inspect_dataset.py.")
    unknown = set(args.relax) - set(rc.get("extra_immutable", []))
    if unknown:
        raise SystemExit(f"--relax names that are not in extra_immutable: {sorted(unknown)}")

    out = resolve_path(args.out)
    ref_out = out.with_name(out.stem + "_reference.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    cols_main, cols_ref = expected_columns(args.heldout)
    _check_header(out, cols_main)               # fail BEFORE spending hours of compute
    _check_header(ref_out, cols_ref)
    done = set()
    if out.exists():
        prev = pd.read_csv(out, usecols=["seed", "variant"])
        done = set(zip(prev["seed"], prev["variant"]))

    for seed in args.seeds:
        if (seed, args.tag) in done:
            print(f"seed {seed}, tag {args.tag}: already in {out.name}, skipped")
            continue
        t0 = time.time()
        df, ref, info = run_seed(seed, args, cfg, ds, rc)
        _append(df, out, cols_main)
        _append(ref, ref_out, cols_ref)
        print(f"seed {seed}, tag {args.tag}: {len(df)} rows in {time.time() - t0:.0f}s ({info}) -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
    