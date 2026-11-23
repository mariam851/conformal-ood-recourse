import numpy as np
import pandas as pd
import pytest

from conformal_trees.data.loaders import TabularDataset
from conformal_trees.evaluation import flag_rate, paired_bootstrap_ci, robust_validity
from conformal_trees.recourse import OODGate, RecourseSpace, SearchConfig, find_recourse, make_donors


class FakeModel:
    """P(y=1) = sigmoid(3 * (x1 + 0.5 * x2 + (cat == 'good') - 2.5))."""

    def predict_proba(self, X):
        z = 3.0 * (X["x1"].to_numpy(float) + 0.5 * X["x2"].to_numpy(float) + (X["cat"].astype(str) == "good") - 2.5)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])


@pytest.fixture
def world():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame(
        {
            "age": rng.integers(20, 70, n),                       # immutable
            "x1": rng.integers(0, 6, n),                          # mutable, integer, monotone up
            "x2": rng.normal(0, 1, n),                            # mutable, continuous
            "gain": np.where(rng.random(n) < 0.9, 0.0, rng.normal(5, 1, n)),   # spike -> not actionable
            "cat": pd.Categorical(rng.choice(["good", "bad", "meh"], n)),
        }
    )
    y = (FakeModel().predict_proba(X)[:, 1] > 0.5).astype(int)
    ds = TabularDataset("toy", X, y, numeric=["age", "x1", "x2", "gain"], categorical=["cat"], immutable=["age"])
    space = RecourseSpace.from_dataset(ds, X, monotone_up=["x1"])
    donors = make_donors(space, X, y)
    return X, y, space, donors


def _cfg(**kw):
    return SearchConfig(pop_size=80, generations=20, **kw)


def _a_negative_row(X):
    p = FakeModel().predict_proba(X)[:, 1]
    return X[(p < 0.2)].iloc[0]


def test_space_flags(world):
    _, _, space, _ = world
    assert space.mutable == ["x1", "x2", "cat"]
    assert space.spiky == ["gain"]
    assert space.integer == ["age", "x1"]
    assert space.monotone_up == ["x1"]


def test_project_enforces_constraints(world):
    X, _, space, _ = world
    orig = space.row_to_dict(X.iloc[0])
    pop = {
        "age": np.array([99.0, 1.0]), "x1": np.array([-5.0, 99.0]), "x2": np.array([1e6, -1e6]),
        "gain": np.array([7.0, 7.0]), "cat": np.array(["good", "bad"], dtype=object),
    }
    out = space.project(pop, orig)
    assert (out["age"] == orig["age"]).all() and (out["gain"] == orig["gain"]).all()      # immutable
    assert (out["x1"] >= orig["x1"]).all() and (out["x1"] <= space.upper["x1"]).all()     # monotone + bounds
    assert (out["x2"] >= space.lower["x2"]).all() and (out["x2"] <= space.upper["x2"]).all()
    assert np.array_equal(out["x1"], np.round(out["x1"]))                                  # integer


def test_cost_and_n_changed(world):
    X, _, space, _ = world
    orig = space.row_to_dict(X.iloc[0])
    pop = space.project({c: np.full(3, orig[c], dtype=object if c == "cat" else float) for c in space.columns}, orig)
    pop["x2"] = pop["x2"].copy(); pop["x2"][1] = orig["x2"] + space.scale["x2"]            # one std
    pop["cat"] = pop["cat"].copy(); pop["cat"][2] = "zzz" if orig["cat"] != "zzz" else "bad"
    assert space.cost(pop, orig).tolist() == pytest.approx([0.0, 1.0, 1.0])
    assert space.n_changed(pop, orig).tolist() == [0, 1, 1]


def test_search_finds_valid_actionable_cheap_counterfactual(world):
    X, _, space, donors = world
    x = _a_negative_row(X)
    res = find_recourse(FakeModel(), space, x, donors, cfg=_cfg(), rng=np.random.default_rng(0))
    assert res.found and res.prob >= 0.5 and res.n_changed >= 1
    assert res.x_cf["age"] == x["age"] and res.x_cf["gain"] == x["gain"]                  # immutable untouched
    assert res.x_cf["x1"] >= x["x1"]                                                       # monotone
    again = find_recourse(FakeModel(), space, x, donors, cfg=_cfg(), rng=np.random.default_rng(0))
    assert again.x_cf == res.x_cf and again.cost == res.cost                               # deterministic


def test_search_reports_failure_when_nothing_is_actionable_or_target_unreachable(world):
    X, y, _, donors = world
    ds = TabularDataset("toy", X, y, numeric=["age", "x1", "x2", "gain"], categorical=["cat"],
                        immutable=["age", "x1", "x2", "cat"])
    frozen = RecourseSpace.from_dataset(ds, X)
    assert frozen.mutable == []
    res = find_recourse(FakeModel(), frozen, _a_negative_row(X), donors, cfg=_cfg(), rng=np.random.default_rng(0))
    assert not res.found and res.x_cf is None
    _, _, space, donors2 = world
    hard = find_recourse(FakeModel(), space, _a_negative_row(X), donors2, cfg=_cfg(target_prob=0.999999),
                         rng=np.random.default_rng(0))
    assert hard.n_evals > 0 and (not hard.found or hard.prob >= 0.999999)


def test_gate_threshold_and_flag_rate():
    rng = np.random.default_rng(1)
    cal = np.abs(rng.normal(size=6000))
    gate = OODGate(lambda X: X["s"].to_numpy(float), cal, alpha=0.05, delta=0.01)
    fresh = pd.DataFrame({"s": np.abs(rng.normal(size=20000))})
    assert 1 - gate.passes(fresh).mean() <= 0.05                                           # FPR <= alpha
    assert gate.passes(pd.DataFrame({"s": [0.1]}))[0] and not gate.passes(pd.DataFrame({"s": [9.0]}))[0]
    assert gate.violation(pd.DataFrame({"s": [0.1]}))[0] == 0.0
    assert gate.violation(pd.DataFrame({"s": [9.0]}))[0] > 0.0
    tiny = OODGate(lambda X: X["s"].to_numpy(float), cal[:20], alpha=0.05, delta=0.01)
    assert tiny.k == 0 and tiny.passes(pd.DataFrame({"s": [1e9]}))[0]                      # n too small: never flags


def test_gate_changes_the_search_result_and_is_always_respected(world):
    """The gate forbids x1 above ~1.9. Without it the cheap path raises x1; with it, never."""
    X, _, space, donors = world
    p = FakeModel().predict_proba(X)[:, 1]
    rows = X[p < 0.2].iloc[:12]
    cal = np.random.default_rng(2).uniform(0, 2.0, 5000)
    gate = OODGate(lambda F: F["x1"].to_numpy(float), cal, alpha=0.05, delta=0.01)
    both, ungated_violates = 0, 0
    for i, (_, x) in enumerate(rows.iterrows()):
        free = find_recourse(FakeModel(), space, x, donors, cfg=_cfg(), rng=np.random.default_rng(i))
        gated = find_recourse(FakeModel(), space, x, donors, cfg=_cfg(), rng=np.random.default_rng(i), gate=gate)
        if free.found and gated.found:
            both += 1
            ungated_violates += free.x_cf["x1"] > gate.score_threshold
            assert gated.x_cf["x1"] <= gate.score_threshold and gated.violation == 0.0
    assert both >= 8
    assert ungated_violates >= 5                       # the gate is not vacuous: it forced a different solution


def test_flag_rate_is_about_alpha_on_id_and_high_on_shifted():
    rng = np.random.default_rng(3)
    cal = rng.normal(size=5000)
    assert abs(flag_rate(cal, rng.normal(size=20000), 0.05) - 0.05) < 0.01
    assert flag_rate(cal, rng.normal(loc=3.0, size=2000), 0.05) > 0.8


def test_robust_validity_and_paired_bootstrap():
    class M:
        def __init__(self, thr): self.thr = thr
        def predict_proba(self, X):
            p = (X["x1"].to_numpy(float) > self.thr).astype(float)
            return np.column_stack([1 - p, p])
    X = pd.DataFrame({"x1": [0.0, 1.5, 3.0]})
    rv = robust_validity([M(0.5), M(1.0), M(2.0)], X, 0.5)
    assert rv.tolist() == pytest.approx([0.0, 2 / 3, 1.0])
    a = np.arange(40, dtype=float) + 1.0
    mean, lo, hi = paired_bootstrap_ci(a, a - 1.0, rng=np.random.default_rng(0))
    assert mean == pytest.approx(1.0) and lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    mean, lo, hi = paired_bootstrap_ci(a, np.zeros(40), rng=np.random.default_rng(0))
    assert lo < mean < hi
    