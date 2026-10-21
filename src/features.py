"""
features.py - turning raw sales into "clues" (features) a model can learn from.

A tree model like XGBoost can't look at a calendar by itself, so we give it
columns such as "day of week", "sales 91 days ago", "average of the last 4
weeks (as of 91 days ago)" and "is it a weekend?".

WHY ALL LAGS ARE >= 91 DAYS:
We want to forecast up to 90 days ahead. On the day we make the forecast we
only know sales up to *today*. If the model used "yesterday's sales" it would
need values we don't have yet when predicting 60 days out. By using only
information at least 91 days old, every feature is available for the whole
forecast window. This is called "direct" forecasting and it prevents leakage.
"""
import numpy as np
import pandas as pd

MAX_HORIZON = 90          # longest forecast the app allows (days)
LAGS = [91, 98, 105, 112, 119, 126, 182, 364, 371, 546, 728]
ROLL_WINDOWS = [7, 28, 91, 182, 364]


def add_calendar_features(df):
    """Calendar columns: day of week, month, weekend and month start/end flags."""
    d = df["date"]
    df["dayofweek"] = d.dt.dayofweek.astype("int8")      # 0 = Monday
    df["month"] = d.dt.month.astype("int8")
    df["dayofmonth"] = d.dt.day.astype("int8")
    df["dayofyear"] = d.dt.dayofyear.astype("int16")
    df["weekofyear"] = d.dt.isocalendar().week.astype("int8").values
    df["year"] = d.dt.year.astype("int16")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
    df["is_month_start"] = d.dt.is_month_start.astype("int8")
    df["is_month_end"] = d.dt.is_month_end.astype("int8")
    return df


def add_lag_features(df):
    """Past-sales columns, computed separately for each store-item series."""
    g = df.groupby(["store", "item"], sort=False)["sales"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag).astype("float32")
    # Rolling averages, all shifted by 91 days so they're known in advance.
    shifted = g.shift(MAX_HORIZON + 1)
    sg = shifted.groupby([df["store"], df["item"]], sort=False)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = sg.transform(lambda s: s.rolling(w, min_periods=1).mean()).astype("float32")
    df["roll_std_28"] = sg.transform(lambda s: s.rolling(28, min_periods=2).std()).astype("float32")
    # Same weekday average: mean of the lag 91..126 values (all same weekday).
    df["same_wday_mean"] = df[[f"lag_{l}" for l in [91, 98, 105, 112, 119, 126]]].mean(axis=1).astype("float32")
    return df


def make_features(df):
    """Main entry point: returns the data with all feature columns added."""
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    return df


def feature_columns(df):
    """Every column except the date and the answer (sales).
    Store and item numbers are kept so one "global" model can learn that
    e.g. item 15 always sells more than item 5."""
    return [c for c in df.columns if c not in ("date", "sales")]
