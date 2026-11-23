import numpy as np
import pandas as pd
import pytest

from conformal_trees.data.ood_scenarios import CovariateShift, build_covariate_shifts, mix_batch

NUMERIC = ["a", "b", "c"]


def test_mean_shift_moves_each_feature_by_severity_times_std(shift_frame):
    s = CovariateShift("mean_shift", 2.0).fit(shift_frame, NUMERIC)
    out = s.apply(shift_frame)
    for col in ("a", "b"):
        assert np.isclose((out[col] - shift_frame[col]).mean(), 2.0 * shift_frame[col].std())


def test_gaussian_noise_has_expected_scale(shift_frame):
    s = CovariateShift("gaussian_noise", 1.5).fit(shift_frame, NUMERIC)
    noise = s.apply(shift_frame, seed=1)["b"] - shift_frame["b"]
    assert abs(noise.mean()) < 1.0
    assert np.isclose(noise.std(), 1.5 * shift_frame["b"].std(), rtol=0.05)


def test_dense_numeric_excludes_spike_features():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame(
        {
            "dense": rng.normal(size=n),
            "spike": np.where(rng.random(n) < 0.9, 0.0, rng.normal(5, 1, n)),   # 90% zeros
        }
    )
    s = CovariateShift("mean_shift", 1.0).fit(X, ["dense", "spike"])             # default: dense_numeric
    assert s.excluded_spiky_ == ["spike"] and set(s.scale_) == {"dense"}
    out = s.apply(X)
    assert (out["spike"] == X["spike"]).all()
    s_all = CovariateShift("mean_shift", 1.0, features="all_numeric").fit(X, ["dense", "spike"])
    assert set(s_all.scale_) == {"dense", "spike"} and s_all.excluded_spiky_ == []


def test_input_not_mutated_and_other_columns_untouched(shift_frame):
    before = shift_frame.copy(deep=True)
    out = CovariateShift("mean_shift", 1.0).fit(shift_frame, NUMERIC).apply(shift_frame)
    pd.testing.assert_frame_equal(shift_frame, before)
    assert list(out.columns) == list(shift_frame.columns)
    assert out["k"].equals(shift_frame["k"])
    assert (out["c"] == shift_frame["c"]).all()


def test_zero_variance_feature_is_skipped(shift_frame):
    s = CovariateShift("mean_shift", 1.0, features="all_numeric").fit(shift_frame, NUMERIC)
    assert s.skipped_ == ["c"]
    assert set(s.scale_) == {"a", "b"}


def test_deterministic_for_same_seed_different_for_other_seed(shift_frame):
    s = CovariateShift("gaussian_noise", 1.0).fit(shift_frame, NUMERIC)
    pd.testing.assert_frame_equal(s.apply(shift_frame, seed=3), s.apply(shift_frame, seed=3))
    assert not s.apply(shift_frame, seed=3).equals(s.apply(shift_frame, seed=4))


def test_statistics_come_from_reference_not_from_shifted_rows(shift_frame):
    ref, rest = shift_frame.iloc[:1000], shift_frame.iloc[1000:]
    s = CovariateShift("mean_shift", 1.0).fit(ref, NUMERIC)
    out = s.apply(rest)
    assert np.isclose((out["a"] - rest["a"]).mean(), ref["a"].std())


def test_integer_valued_features_stay_integer_valued():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"n": rng.integers(0, 100, 500), "x": rng.normal(size=500)})
    for kind in ("mean_shift", "gaussian_noise"):
        s = CovariateShift(kind, 0.37).fit(X, ["n", "x"])
        out = s.apply(X, seed=1)
        assert s.integer_features_ == ["n"]
        assert (out["n"] == np.round(out["n"])).all()      # no fractional-part artifact
        assert not (out["x"] == np.round(out["x"])).all()  # continuous feature untouched by rounding


def test_missing_values_survive_the_shift():
    X = pd.DataFrame({"n": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]})
    out = CovariateShift("mean_shift", 1.0).fit(X, ["n"]).apply(X)
    assert out["n"].isna().tolist() == [False, False, True, False, False, False]


def test_explicit_feature_subset(shift_frame):
    s = CovariateShift("mean_shift", 1.0, features=["a"]).fit(shift_frame, NUMERIC)
    out = s.apply(shift_frame)
    assert (out["b"] == shift_frame["b"]).all()
    assert not (out["a"] == shift_frame["a"]).any()


def test_invalid_arguments(shift_frame):
    with pytest.raises(ValueError):
        CovariateShift("nonsense", 1.0)
    with pytest.raises(ValueError):
        CovariateShift("mean_shift", 0.0)
    with pytest.raises(ValueError):
        CovariateShift("mean_shift", 1.0, features=["k"]).fit(shift_frame, NUMERIC)  # categorical
    with pytest.raises(ValueError):
        CovariateShift("mean_shift", 1.0, features=["c"]).fit(shift_frame, NUMERIC)  # zero variance
    with pytest.raises(RuntimeError):
        CovariateShift("mean_shift", 1.0).apply(shift_frame)                          # not fitted


def test_build_grid(shift_frame):
    cfg = {"covariate_shift": {"kinds": ["mean_shift", "gaussian_noise"], "severities": [0.5, 1.0, 2.0]}}
    shifts = build_covariate_shifts(cfg, shift_frame, NUMERIC)
    assert len(shifts) == 6
    assert len({s.name for s in shifts}) == 6



def test_mix_batch_contamination_and_provenance(shift_frame):
    X_ood = CovariateShift("mean_shift", 1.0).fit(shift_frame, NUMERIC).apply(shift_frame)
    batch, is_ood = mix_batch(shift_frame, X_ood, 0.3, seed=1)
    assert is_ood.sum() == round(0.3 * len(shift_frame))
    assert np.allclose(batch.loc[is_ood, "a"], X_ood.loc[is_ood, "a"])
    assert np.allclose(batch.loc[~is_ood, "a"], shift_frame.loc[~is_ood, "a"])
    assert batch["k"].equals(shift_frame["k"])                        # untouched column preserved
    again, is_ood2 = mix_batch(shift_frame, X_ood, 0.3, seed=1)
    assert (is_ood == is_ood2).all() and again.equals(batch)          # deterministic
    assert not (mix_batch(shift_frame, X_ood, 0.3, seed=2)[1] == is_ood).all()


def test_mix_batch_extremes_and_validation(shift_frame):
    X_ood = CovariateShift("mean_shift", 1.0).fit(shift_frame, NUMERIC).apply(shift_frame)
    none, m0 = mix_batch(shift_frame, X_ood, 0.0)
    allo, m1 = mix_batch(shift_frame, X_ood, 1.0)
    assert not m0.any() and m1.all()
    assert np.allclose(none["a"], shift_frame["a"]) and np.allclose(allo["a"], X_ood["a"])
    with pytest.raises(ValueError):
        mix_batch(shift_frame, X_ood, 1.5)
    with pytest.raises(ValueError):
        mix_batch(shift_frame, X_ood.iloc[:10], 0.5)



def test_mix_batch_contamination_and_provenance(shift_frame):
    X_ood = CovariateShift("mean_shift", 1.0).fit(shift_frame, NUMERIC).apply(shift_frame)
    batch, is_ood = mix_batch(shift_frame, X_ood, 0.3, seed=1)
    assert is_ood.sum() == round(0.3 * len(shift_frame))
    assert np.allclose(batch.loc[is_ood, "a"], X_ood.loc[is_ood, "a"])
    assert np.allclose(batch.loc[~is_ood, "a"], shift_frame.loc[~is_ood, "a"])
    assert batch["k"].equals(shift_frame["k"])                        # untouched column preserved
    again, is_ood2 = mix_batch(shift_frame, X_ood, 0.3, seed=1)
    assert (is_ood == is_ood2).all() and again.equals(batch)          # deterministic
    assert not (mix_batch(shift_frame, X_ood, 0.3, seed=2)[1] == is_ood).all()


def test_mix_batch_extremes_and_validation(shift_frame):
    X_ood = CovariateShift("mean_shift", 1.0).fit(shift_frame, NUMERIC).apply(shift_frame)
    none, m0 = mix_batch(shift_frame, X_ood, 0.0)
    allo, m1 = mix_batch(shift_frame, X_ood, 1.0)
    assert not m0.any() and m1.all()
    assert np.allclose(none["a"], shift_frame["a"]) and np.allclose(allo["a"], X_ood["a"])
    with pytest.raises(ValueError):
        mix_batch(shift_frame, X_ood, 1.5)
    with pytest.raises(ValueError):
        mix_batch(shift_frame, X_ood.iloc[:10], 0.5)
