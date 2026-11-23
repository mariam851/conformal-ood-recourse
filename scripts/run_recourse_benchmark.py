"""Multi-seed recourse benchmark: one CSV row per (seed, target, gate, individual).

Compares the same evolutionary search with no gate and with OOD gates built on different detectors
(Isolation Forest, LOF). Every counterfactual is then scored by ALL detectors (plain alpha-level conformal test),
so a gate's effect on its own detector and on the others can be separated. Also stores reference rows.

Usage (from the project root):
    python scripts/run_recourse_benchmark.py --seeds 0 1 2 --n 200 --targets 0.5 0.6 0.7
Outputs (resumable: seeds already in the file are skipped):
    results/raw/recourse_benchmark.csv             per-individual results
    results/raw/recourse_benchmark_reference.csv   detector flag rates of original / real favourable rows
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from conformal_trees.conformal import conformal_pvalues
from conformal_trees.data import load_dataset, make_splits
from conformal_trees.evaluation import robust_validity
from conformal_trees.models import GBDTClassifier
from conformal_trees.recourse import OODGate, RecourseSpace, SearchConfig, find_recourse, make_donors
from conformal_trees.signals import IsolationForestScore, KNNScore, LOFScore
from conformal_trees.utils import derive_seed, load_config, load_dataset_config, make_rng, resolve_path


def run_seed(seed: int, args, cfg, ds, rc) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha, delta = cfg["gate"]["alpha"], cfg["gate"]["delta"]
    sp = make_splits(ds.X, ds.y, cfg["splits"], seed=seed)
    model = GBDTClassifier.from_config(cfg["gbdt"], seed=seed).fit(sp.train.X, sp.train.y)
    alt = [GBDTClassifier.from_config(cfg["gbdt"], seed=derive_seed(seed, "alt", i)).fit(sp.train.X, sp.train.y)
           for i in range(args.n_alt)]
    dets = {
        "IF": IsolationForestScore(ds.numeric, ds.categorical, seed=seed).fit(sp.train.X),
        "LOF": LOFScore(ds.numeric, ds.categorical).fit(sp.train.X),
        "kNN": KNNScore(ds.numeric, ds.categorical).fit(sp.train.X),
    }
    cal = {k: d.score(sp.calib.X) for k, d in dets.items()}
    all_gates = {
        "none": None,
        "gate_IF": OODGate(dets["IF"].score, cal["IF"], alpha, delta),
        "gate_LOF": OODGate(dets["LOF"].score, cal["LOF"], alpha, delta),
    }
    gates = {k: v for k, v in all_gates.items() if k in args.gates}

    space = RecourseSpace.from_dataset(
        ds, sp.train.X, extra_immutable=rc["extra_immutable"], monotone_up=rc["monotone_increasing"],
        spike_immutable=cfg["recourse"]["spike_immutable"], max_mode_share=cfg["recourse"]["max_mode_share"],
    )
    donors = make_donors(space, sp.train.X, sp.train.y)
    search_cfg = SearchConfig.from_config(cfg)

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
                rows.append({"seed": seed, "target": t, "gate": gname, "i": i, "idx": int(idx), "found": r.found,
                             "prob": r.prob, "cost": r.cost if r.found else np.nan,
                             "n_changed": r.n_changed if r.found else np.nan, **extra.get(i, {})})

    real_pos = sp.test.X[p_test >= 0.5]
    ref = []
    for label, Xr in (("original_unfavourable", sp.test.X.iloc[picked]), ("real_favourable", real_pos)):
        fl = flags(Xr)
        ref.append({"seed": seed, "group": label, "n": len(Xr), **{f"flag_{k}": float(v.mean()) for k, v in fl.items()}})
    return pd.DataFrame(rows), pd.DataFrame(ref)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n", type=int, default=200, help="individuals per seed")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.5, 0.6, 0.7])
    ap.add_argument("--gates", nargs="+", default=["none", "gate_IF", "gate_LOF"],
                    choices=["none", "gate_IF", "gate_LOF"])
    ap.add_argument("--n-alt", type=int, default=3, help="re-trained models used for the robustness check")
    ap.add_argument("--out", default="results/raw/recourse_benchmark.csv")
    args = ap.parse_args()

    cfg = load_config("base.yaml", "models.yaml", "recourse.yaml")
    ds = load_dataset("adult")
    rc = load_dataset_config("adult")["recourse"]
    out = resolve_path(args.out)
    ref_out = out.with_name(out.stem + "_reference.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out, usecols=["seed"])["seed"]) if out.exists() else set()

    for seed in args.seeds:
        if seed in done:
            print(f"seed {seed}: already in {out.name}, skipped")
            continue
        t0 = time.time()
        df, ref = run_seed(seed, args, cfg, ds, rc)
        df.to_csv(out, mode="a", header=not out.exists(), index=False)
        ref.to_csv(ref_out, mode="a", header=not ref_out.exists(), index=False)
        print(f"seed {seed}: {len(df)} rows in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
    