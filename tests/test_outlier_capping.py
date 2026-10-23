import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import cap_outliers, clean_sales  # noqa: E402


def _df(n=200):
    r = np.random.default_rng(1)
    d = pd.date_range("2024-01-01", periods=n)
    return clean_sales(pd.DataFrame({"date": d, "item": 1, "sales": 100 + r.normal(0, 3, n).round()}))


def test_spike_is_capped_not_deleted():
    df = _df()
    df.loc[100, "sales"] = 5000
    out = cap_outliers(df)
    assert len(out) == len(df)
    assert 100 < out.loc[100, "sales"] < 200


def test_normal_days_unchanged():
    df = _df()
    assert (cap_outliers(df)["sales"] == df["sales"]).all()


def test_raw_input_not_modified():
    df = _df()
    df.loc[50, "sales"] = 5000
    cap_outliers(df)
    assert df.loc[50, "sales"] == 5000
