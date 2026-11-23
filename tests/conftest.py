import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_xy():
    rng = np.random.default_rng(123)
    n = 5000
    y = (rng.random(n) < 0.25).astype(int)          # imbalanced, like Adult
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 10, size=n)})
    return X, y


@pytest.fixture
def shift_frame():
    rng = np.random.default_rng(0)
    n = 4000
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(50, 10, n),
            "c": np.zeros(n),                        # zero variance -> must be skipped
            "k": pd.Categorical(rng.choice(["x", "y"], n)),
        }
    )
