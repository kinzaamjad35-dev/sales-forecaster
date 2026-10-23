"""Robustness tests for messy uploaded CSVs."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import clean_sales, load_sales  # noqa: E402
from src.features import make_features  # noqa: E402
from src import models as M  # noqa: E402


def _series(days, start="2023-01-01"):
    d = pd.date_range(start, periods=days)
    return pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "item": 1, "sales": 50 + np.arange(days) % 7})


def test_day_first_dates_are_not_swapped():
    df = _series(60)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%d/%m/%Y")
    out = clean_sales(df)
    assert len(out) == 60
    assert out["date"].min() == pd.Timestamp("2023-01-01")


def test_blank_sales_are_interpolated_not_zero():
    df = _series(10)
    df.loc[4, "sales"] = np.nan
    out = clean_sales(df)
    assert out.loc[4, "sales"] > 0


def test_absent_days_still_count_as_zero():
    df = _series(10).drop(index=4)
    assert clean_sales(df).loc[4, "sales"] == 0


def test_column_aliases_and_comma_numbers():
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Product": ["A", "A"], "Qty": ["1,200", "Rs 300"]})
    out = clean_sales(df)
    assert list(out["sales"]) == [1200, 300]


def test_text_item_ids_become_numbers():
    df = _series(5)
    df["item"] = "Shirt"
    assert pd.api.types.is_numeric_dtype(clean_sales(df)["item"])


def test_empty_csv_gives_clear_error():
    with pytest.raises(ValueError, match="empty|No usable"):
        load_sales(io.StringIO("date,item,sales\n"))


def test_short_history_trains_or_explains():
    feat = make_features(clean_sales(_series(200)))
    M.train_lightgbm(feat, n_estimators=10)  # under a year: falls back to 91-day lag rows
    with pytest.raises(ValueError, match="Not enough history"):
        M.train_lightgbm(make_features(clean_sales(_series(30))), n_estimators=10)
