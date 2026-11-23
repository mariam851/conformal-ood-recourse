"""Tables and figures for the paper, built from the benchmark CSVs.

Everything here is a pure function of the CSV files written by scripts/run_recourse_benchmark.py and
scripts/run_gate_benchmark.py, so the paper artifacts can always be regenerated:

    python scripts/make_paper_artifacts.py

Conventions
-----------
* "Per method" numbers are means over seeds of per-seed means (sd across seeds is the conservative uncertainty).
* "Paired" numbers use only individuals for which both the gate and the no-gate search found a counterfactual
  (a gate can look better just by failing on hard cases). Confidence intervals pool individuals over seeds and
  treat them as independent, so they are optimistic; the exact McNemar test and the per-seed consistency count
  are reported next to them. Holm correction is applied across all flag comparisons in a table.
* Verdicts (the pre-registered rule, docs/decisions.md D21/D22) for detectors a gate was NOT built on:
    transfer     : diff <= -0.02, CI entirely below 0 (and Holm p < 0.05 when require_holm=True)
    none         : |diff| < 0.01 and CI contains 0
    inconclusive : anything else
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from conformal_trees.evaluation.stats import paired_bootstrap_ci
from conformal_trees.utils.seed import derive_seed, make_rng

BASE_METRICS = ["cost", "n_changed"]
BUILT_ON = {"gate_IF": {"IF"}, "gate_LOF": {"LOF"}, "gate_kNN": {"kNN"}, "gate_ALL": {"IF", "LOF", "kNN"}}
POOL = {"IF", "LOF", "kNN"}
ROLE_COLORS = {"own": "#c0392b", "pool (not in gate)": "#2471a3", "held-out": "#1e8449"}


# --------------------------------------------------------------------------------------------- loading
def flag_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("flag_")]


def load_benchmark(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "variant" not in df:
        df["variant"] = "main"
    if "dataset" not in df:
        df["dataset"] = "adult"
    df["found"] = df["found"].astype(bool)
    return df


# --------------------------------------------------------------------------------------------- summaries
def per_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in BASE_METRICS + flag_columns(df) + ["robust"] if m in df]
    agg = {"success": ("found", "mean"), **{m: (m, "mean") for m in metrics}}
    return df.groupby(["variant", "seed", "target", "gate"]).agg(**agg).reset_index()


def summary_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(mean, sd) across seeds of the per-seed means, indexed by (variant, target, gate)."""
    ps = per_seed_summary(df)
    cols = [c for c in ps.columns if c not in ("variant", "seed", "target", "gate")]
    g = ps.groupby(["variant", "target", "gate"])[cols]
    return g.mean(), g.std()


def summary_strings(mean: pd.DataFrame, sd: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = mean.copy().astype(object)
    for c in mean.columns:
        for idx in mean.index:
            m, s = mean.loc[idx, c], sd.loc[idx, c]
            out.loc[idx, c] = f"{m:.{digits}f}" if pd.isna(s) else f"{m:.{digits}f} ± {s:.{digits}f}"
    return out


# --------------------------------------------------------------------------------------------- paired statistics
def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, j in enumerate(order):
        running = max(running, (len(order) - rank) * p[j])
        adj[j] = min(1.0, running)
    return adj


def _pair(df, gate, target, metric, variant):
    d = df[(df.target == target) & df.found]
    if variant is not None:
        d = d[d.variant == variant]
    key = ["variant", "seed", "i"]
    a = d[d.gate == gate].set_index(key)[metric]
    b = d[d.gate == "none"].set_index(key)[metric]
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def paired_effects(df, gates, targets, metrics, *, variant=None, seed=0, n_boot=2000) -> pd.DataFrame:
    """gate minus none, on individuals found by both. Binary flag metrics also get an exact McNemar test."""
    rng = make_rng(derive_seed(seed, "paper", "paired"))
    rows = []
    for gate in gates:
        for t in targets:
            for m in metrics:
                a, b = _pair(df, gate, t, m, variant)
                a, b = a[a.notna() & b.notna()], b[a.notna() & b.notna()]
                if len(a) == 0:
                    continue
                mean, lo, hi = paired_bootstrap_ci(a, b, rng=rng, n_boot=n_boot)
                row = {"gate": gate, "target": t, "metric": m, "diff": mean, "ci_low": lo, "ci_high": hi,
                       "n_pairs": len(a), "only_gate": np.nan, "only_none": np.nan, "p": np.nan,
                       "seeds_drop_2pts": np.nan, "n_seeds": np.nan}
                if m.startswith("flag_"):
                    og, on = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
                    per_seed = (a - b).groupby(level="seed").mean()
                    row.update(only_gate=og, only_none=on,
                               p=binomtest(og, og + on, 0.5).pvalue if og + on else np.nan,
                               seeds_drop_2pts=int((per_seed <= -0.02).sum()), n_seeds=int(per_seed.size))
                rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].to_numpy())
    return out


def add_roles_and_verdicts(effects: pd.DataFrame, *, require_holm: bool = False) -> pd.DataFrame:
    out = effects.copy()

    def role(r):
        if not str(r["metric"]).startswith("flag_"):
            return ""
        d = r["metric"][len("flag_"):]
        if d in BUILT_ON.get(r["gate"], set()):
            return "own"
        return "pool (not in gate)" if d in POOL else "held-out"

    out["role"] = out.apply(role, axis=1)

    def verdict(r):
        if r["role"] in ("", "own"):
            return ""
        holm_ok = (not require_holm) or (r["p_holm"] < 0.05)
        if r["diff"] <= -0.02 and r["ci_high"] < 0 and holm_ok:
            return "transfer"
        if abs(r["diff"]) < 0.01 and r["ci_low"] <= 0 <= r["ci_high"]:
            return "none"
        return "inconclusive"

    out["verdict"] = out.apply(verdict, axis=1)
    return out


# --------------------------------------------------------------------------------------------- LaTeX
_ESC = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}


def _tex(s) -> str:
    s = str(s)
    for k, v in _ESC.items():
        s = s.replace(k, v)
    return s.replace("±", r"$\pm$")


def to_latex(df: pd.DataFrame, path, caption: str, label: str, float_fmt: str = "{:.3f}") -> Path:
    t = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{_tex(caption)}}}", rf"\label{{{label}}}",
             r"\small", rf"\begin{{tabular}}{{{'l' * len(t.columns)}}}", r"\toprule",
             " & ".join(_tex(c) for c in t.columns) + r" \\", r"\midrule"]
    for _, r in t.iterrows():
        cells = [float_fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                 else ("" if (isinstance(v, float) and pd.isna(v)) else _tex(v)) for v in r.tolist()]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------------- figures
def _save(fig, stem) -> list[Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ("png", "pdf"):
        f = Path(f"{stem}.{ext}")
        fig.savefig(f, dpi=200, bbox_inches="tight")
        files.append(f)
    return files


def fig_margin_tradeoff(df: pd.DataFrame, stem=None, variant: str = "main"):
    """success, robust and cost versus the search target; mean +- sd over seeds, one line per gate."""
    import matplotlib.pyplot as plt

    ps = per_seed_summary(df[df.variant == variant])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (col, title) in zip(axes, [("success", "success rate"), ("robust", "robustness to re-training"),
                                       ("cost", "cost (std units)")]):
        for gate, g in ps.groupby("gate"):
            s = g.groupby("target")[col].agg(["mean", "std"]).fillna(0.0)
            ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, label=gate)
        ax.set_xlabel("search target P(y=1)")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    if stem is not None:
        _save(fig, stem)
    return fig


def fig_paired_effects(effects: pd.DataFrame, target: float, stem=None, title: str | None = None):
    """Paired flag-rate change (gate - none) with 95% CI, coloured by the role of the detector."""
    import matplotlib.pyplot as plt

    e = effects[(effects.target == target) & effects.metric.str.startswith("flag_") & (effects.role != "")]
    labels = [f"{r.gate} -> {r.metric[5:]}" for r in e.itertuples()]
    fig, ax = plt.subplots(figsize=(7, 0.32 * max(len(e), 3) + 1.2))
    y = np.arange(len(e))[::-1]
    for yi, r in zip(y, e.itertuples()):
        ax.errorbar(r.diff, yi, xerr=[[r.diff - r.ci_low], [r.ci_high - r.diff]], fmt="o", capsize=3,
                    color=ROLE_COLORS.get(r.role, "gray"))
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(-0.02, color="gray", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("paired change in flag rate (gate - none)")
    ax.set_title(title or f"Effect of each gate on each detector (target {target:g})", fontsize=10)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=k) for k, c in ROLE_COLORS.items()
               if k in set(e.role)]
    ax.legend(handles=handles, fontsize=8, loc="best")
    fig.tight_layout()
    if stem is not None:
        _save(fig, stem)
    return fig


def fig_flags_vs_reference(df: pd.DataFrame, ref: pd.DataFrame, target: float, stem=None, variant: str = "main"):
    """Flag rate per detector for every gate, against real favourable rows (the right reference)."""
    import matplotlib.pyplot as plt

    ps = per_seed_summary(df[(df.variant == variant) & (df.target == target)])
    dets = [c for c in flag_columns(ps)]
    gates = list(ps.gate.unique())
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.8 / max(len(gates), 1)
    x = np.arange(len(dets))
    for k, gate in enumerate(gates):
        g = ps[ps.gate == gate]
        ax.bar(x + k * w, [g[d].mean() for d in dets], w, yerr=[g[d].std() if len(g) > 1 else 0 for d in dets],
               capsize=2, label=gate)
    real = ref[(ref.variant == variant) & (ref.group == "real_favourable")][dets].mean()
    ax.scatter(x + 0.4 - w / 2, real.values, marker="_", s=900, color="k", label="real favourable rows", zorder=3)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([d[5:] for d in dets])
    ax.set_ylabel("flag rate at alpha = 0.05")
    ax.set_title(f"Counterfactual flag rate by detector (target {target:g})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if stem is not None:
        _save(fig, stem)
    return fig


# --------------------------------------------------------------------------------------------- orchestration
def _gates_of(df):
    return [g for g in df.gate.unique() if g != "none"]


def make_all(bench_csv=None, gate_csv=None, tables_dir="results/tables", figures_dir="paper/figures",
             headline_target: float = 0.6) -> dict[str, list[Path]]:
    """Build every table (CSV + LaTeX) and figure (PNG + PDF) that the available CSVs allow."""
    import matplotlib.pyplot as plt

    tables_dir, figures_dir = Path(tables_dir), Path(figures_dir)
    made: dict[str, list[Path]] = {}

    def table(name, df, caption, float_fmt="{:.3f}"):
        tables_dir.mkdir(parents=True, exist_ok=True)
        csv = tables_dir / f"{name}.csv"
        (df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df).to_csv(csv, index=False)
        made[name] = [csv, to_latex(df, tables_dir / f"{name}.tex", caption, f"tab:{name}", float_fmt)]

    def figure(name, fig):
        made[name] = _save(fig, figures_dir / name)
        plt.close(fig)

    for tag, csv_path, require_holm in (("benchmark", bench_csv, False), ("gate_benchmark", gate_csv, True)):
        if csv_path is None or not Path(csv_path).exists():
            continue
        df = load_benchmark(csv_path)
        ref_path = Path(csv_path).with_name(Path(csv_path).stem + "_reference.csv")
        ref = load_benchmark_ref(ref_path) if ref_path.exists() else None

        mean, sd = summary_tables(df)
        table(f"{tag}_summary", summary_strings(mean, sd),
              f"Mean $\\pm$ sd over seeds ({tag.replace('_', ' ')}).")

        targets = sorted(df.target.unique())
        t_head = headline_target if headline_target in targets else targets[-1]
        metrics = flag_columns(df) + ["robust", "cost", "n_changed"]
        effects = add_roles_and_verdicts(
            paired_effects(df, _gates_of(df), targets, metrics), require_holm=require_holm)
        table(f"{tag}_paired_effects", effects.round(4),
              f"Paired effect of each gate versus no gate ({tag.replace('_', ' ')}).", "{:.4f}")

        figure(f"{tag}_margin_tradeoff", fig_margin_tradeoff(df))
        figure(f"{tag}_paired_effects", fig_paired_effects(effects, t_head))
        if ref is not None:
            figure(f"{tag}_flags_vs_reference", fig_flags_vs_reference(df, ref, t_head))
    return made


def load_benchmark_ref(path) -> pd.DataFrame:
    ref = pd.read_csv(path)
    if "variant" not in ref:
        ref["variant"] = "main"
    return ref
