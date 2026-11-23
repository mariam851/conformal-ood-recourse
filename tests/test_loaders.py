import numpy as np
import pandas as pd
import pytest

from conformal_trees.data.loaders import UNKNOWN, load_dataset
from conformal_trees.utils.seed import derive_seed

CFG = {
    "name": "toy",
    "raw_file": "toy.csv",
    "target": "label",
    "positive_label": ">50K",
    "drop_columns": ["junk"],
    "numeric": ["age", "hours"],
    "categorical": ["job"],
    "immutable": ["age"],
}


@pytest.fixture
def raw_dir(tmp_path):
    df = pd.DataFrame(
        {
            "age": [25, 40, 33, 51, 29, 61],
            "hours": [40, 50, 20, 45, "?", 38],
            "job": ["a", "?", "b", "a", "b", " a "],
            "junk": [1, 2, 3, 4, 5, 6],
            "label": [">50K.", "<=50K.", "<=50K.", ">50K.", "<=50K.", "<=50K."],
        }
    )
    df.to_csv(tmp_path / "toy.csv", index=False)
    return tmp_path


def test_labels_and_cleaning(raw_dir):
    ds = load_dataset("toy", dataset_cfg=CFG, raw_dir=raw_dir)
    assert ds.y.tolist() == [1, 0, 0, 1, 0, 0]          # trailing dots handled
    assert list(ds.X.columns) == ["age", "hours", "job"]  # junk + target removed
    assert pd.isna(ds.X.loc[4, "hours"])                  # "?" numeric -> NaN
    assert ds.X.loc[1, "job"] == UNKNOWN                  # "?" categorical -> Unknown
    assert ds.X.loc[5, "job"] == "a"                      # whitespace stripped
    assert str(ds.X["job"].dtype) == "category"
    assert ds.mutable == ["hours", "job"]


def test_undeclared_column_is_an_error(raw_dir):
    cfg = {**CFG, "drop_columns": []}                     # 'junk' now undeclared
    with pytest.raises(ValueError, match="neither declared nor dropped"):
        load_dataset("toy", dataset_cfg=cfg, raw_dir=raw_dir)


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_dataset("toy", dataset_cfg=CFG, raw_dir=tmp_path)


def test_derive_seed_is_stable_and_distinct():
    assert derive_seed(0, "split", "test") == derive_seed(0, "split", "test")
    assert derive_seed(0, "split", "test") != derive_seed(0, "split", "calib")
    assert derive_seed(0, "x") != derive_seed(1, "x")
    assert 0 <= derive_seed(5, "x") < 2**32
    