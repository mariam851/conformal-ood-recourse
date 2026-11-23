import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from conformal_trees.evaluation import paper_artifacts as pa  # noqa: E402

DETS = ["IF", "LOF", "kNN", "GMM", "OCSVM"]


def _synthetic(seed=0, seeds=3, n=300, targets=(0.5, 0.6)):
    """gate_IF lowers IF strongly, kNN a bit and nothing else; gate_ALL also lowers GMM (held-out)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(seeds):
        for t in targets:
            base = {d: (rng.random(n) < p).astype(float) for d, p in zip(DETS, [.09, .08, .11, .10, .07])}
            for gate in ("none", "gate_IF", "gate_LOF", "gate_ALL"):
                f = {d: v.copy() for d, v in base.items()}
                if gate == "gate_IF":
                    f["IF"] *= rng.random(n) < .25
                    f["kNN"] *= rng.random(n) < .85
                if gate == "gate_LOF":
                    f["LOF"] *= rng.random(n) < .3
                if gate == "gate_ALL":
                    for d in ("IF", "LOF", "kNN"):
                        f[d] *= rng.random(n) < .25
                    f["GMM"] *= rng.random(n) < .4
                found = rng.random(n) < (0.55 if gate != "none" else 0.6)
                for i in range(n):
                    row = {"dataset": "adult", "seed": s, "variant": "main", "target": t, "gate": gate, "i": i,
                           "idx": i, "found": bool(found[i]), "prob": 0.6, "cost": 2.5 + 0.02 * (gate != "none"),
                           "n_changed": 2.0, "robust": float(rng.random() < 0.9)}
                    for d in DETS:
                        row[f"flag_{d}"] = f[d][i]
                    if not found[i]:
                        row.update({k: np.nan for k in ["cost", "n_changed", "robust", *[f"flag_{d}" for d in DETS]]})
                    rows.append(row)
    return pd.DataFrame(rows)


def _reference(seeds=3):
    rows = []
    for s in range(seeds):
        for g, vals in (("real_favourable", [.07, .05, .06, .06, .05]), ("original_unfavourable", [.05, .07, .05, .05, .05])):
            rows.append({"dataset": "adult", "seed": s, "variant": "main", "group": g, "n": 500,
                         **{f"flag_{d}": v for d, v in zip(DETS, vals)}})
    return pd.DataFrame(rows)


def test_holm_matches_textbook_and_handles_nan():
    p = np.array([0.01, 0.04, np.nan, 0.03])
    out = pa.holm(p)
    assert np.isnan(out[2])
    assert out[0] == pytest.approx(0.03) and out[3] == pytest.approx(0.06) and out[1] == pytest.approx(0.06)
    assert (out[~np.isnan(out)] >= p[~np.isnan(p)] - 1e-12).all()


def test_paired_effects_recover_planted_effects_and_verdicts():
    df = _synthetic()
    metrics = [f"flag_{d}" for d in DETS] + ["robust", "cost"]
    eff = pa.add_roles_and_verdicts(pa.paired_effects(df, ["gate_IF", "gate_LOF", "gate_ALL"], [0.6], metrics),
                                    require_holm=True)
    get = lambda g, m: eff[(eff.gate == g) & (eff.metric == m)].iloc[0]
    assert get("gate_IF", "flag_IF").role == "own" and get("gate_IF", "flag_IF")["diff"] < -0.04
    assert get("gate_IF", "flag_LOF").verdict == "none"
    assert get("gate_LOF", "flag_IF").verdict == "none"
    assert get("gate_ALL", "flag_GMM").role == "held-out" and get("gate_ALL", "flag_GMM").verdict == "transfer"
    assert get("gate_IF", "flag_GMM").verdict == "none"
    assert get("gate_ALL", "flag_IF").role == "own"
    assert get("gate_IF", "robust").role == "" and get("gate_IF", "cost")["diff"] == pytest.approx(0.02)
    assert (eff.dropna(subset=["n_pairs"]).n_pairs > 100).all()


def test_roles_distinguish_pool_and_held_out():
    eff = pd.DataFrame([{"gate": "gate_IF", "target": .6, "metric": m, "diff": 0.0, "ci_low": -.01, "ci_high": .01,
                         "p": 1.0, "p_holm": 1.0} for m in ("flag_IF", "flag_LOF", "flag_GMM", "robust")])
    out = pa.add_roles_and_verdicts(eff)
    assert out.role.tolist() == ["own", "pool (not in gate)", "held-out", ""]
    assert out.verdict.tolist() == ["", "none", "none", ""]


def test_summary_tables_shapes_and_strings():
    df = _synthetic()
    mean, sd = pa.summary_tables(df)
    assert mean.index.names == ["variant", "target", "gate"] and mean.shape[0] == 8
    assert {"success", "cost", "robust", "flag_IF", "flag_OCSVM"} <= set(mean.columns)
    s = pa.summary_strings(mean, sd)
    assert "±" in s.iloc[0]["success"]
    one_seed = pa.summary_strings(*pa.summary_tables(df[df.seed == 0]))
    assert "±" not in one_seed.iloc[0]["success"]            # sd undefined with one seed: show the mean only


def test_latex_writer_escapes_and_has_structure(tmp_path):
    df = pd.DataFrame({"gate_x": ["a_b"], "v": [0.12345]})
    path = pa.to_latex(df, tmp_path / "t.tex", caption="100% test", label="tab:t")
    txt = path.read_text(encoding="utf-8")
    assert r"gate\_x" in txt and r"a\_b" in txt and r"100\% test" in txt and "0.123" in txt
    assert txt.count(r"\toprule") == 1 and txt.strip().endswith(r"\end{table}")


def test_make_all_builds_tables_and_figures(tmp_path):
    bench = tmp_path / "bench.csv"
    _synthetic().to_csv(bench, index=False)
    _reference().to_csv(tmp_path / "bench_reference.csv", index=False)
    made = pa.make_all(bench_csv=bench, gate_csv=None, tables_dir=tmp_path / "tables", figures_dir=tmp_path / "figs")
    assert {"benchmark_summary", "benchmark_paired_effects", "benchmark_margin_tradeoff",
            "benchmark_paired_effects", "benchmark_flags_vs_reference"} <= set(made)
    for files in made.values():
        for f in files:
            assert f.exists() and f.stat().st_size > 0
    assert any(f.suffix == ".pdf" for f in made["benchmark_margin_tradeoff"])
    assert pa.make_all(bench_csv=tmp_path / "missing.csv", tables_dir=tmp_path / "t2", figures_dir=tmp_path / "f2") == {}
    