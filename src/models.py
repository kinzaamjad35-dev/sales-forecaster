"""
models.py - the forecasting models and how we score them.

Models (from simplest to smartest):
  1. Naive (last year)  : "sales will equal the same day 52 weeks ago"
  2. Moving average     : "sales will equal the average of the last 28 days"
  3. Prophet            : Facebook/Meta's trend + seasonality model (one per series)
  4. XGBoost / LightGBM : gradient-boosted decision trees using our features

Baselines matter: a fancy model is only useful if it beats the simple ones.

Metrics:
  MAE   = average absolute error in units ("we're off by ~6 units a day")
  MAPE  = average error as a % of actual sales (skips days with 0 sales)
  SMAPE = symmetric % error, robust when sales are small
"""
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from joblib import Parallel, delayed

from .features import feature_columns, MAX_HORIZON

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


# ---------------------------------------------------------------- metrics
def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae = np.mean(np.abs(y_true - y_pred))
    nz = y_true != 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100
    return {"MAE": mae, "MAPE %": mape, "SMAPE %": smape}


# ---------------------------------------------------------------- split
def time_split(df, test_days=MAX_HORIZON):
    """Train = everything before the last `test_days` days, test = the rest.
    We never shuffle time series: the model must not peek at the future."""
    cutoff = df["date"].max() - pd.Timedelta(days=test_days - 1)
    return df[df["date"] < cutoff], df[df["date"] >= cutoff], cutoff


# ---------------------------------------------------------------- baselines
def naive_last_year(test):
    """Same day 364 days earlier (364 = 52 weeks, so the weekday matches)."""
    return test["lag_364"].values


def moving_average(train, test, window=28):
    """Average of each series' last `window` training days, repeated forward."""
    last = (train.sort_values("date").groupby(["store", "item"])["sales"]
            .apply(lambda s: s.tail(window).mean()).rename("ma"))
    return test.join(last, on=["store", "item"])["ma"].values


# ---------------------------------------------------------------- tree models
LGB_PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=63,
                  min_child_samples=50, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, verbose=-1, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=8,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1)


def _clean_train(train):
    # The first ~2 years have no 728-day lag; trees handle NaN, but rows with
    # no history at all are useless, so we drop rows missing the 364-day lag.
    return train.dropna(subset=["lag_364"])


def train_lightgbm(train, objective="regression", alpha=None, **kw):
    """objective="quantile" with alpha=0.1/0.9 gives lower/upper bounds,
    which we use as an 80% prediction interval."""
    tr = _clean_train(train)
    params = {**LGB_PARAMS, **kw, "objective": objective}
    if alpha is not None:
        params["alpha"] = alpha
    model = lgb.LGBMRegressor(**params)
    model.fit(tr[feature_columns(tr)], tr["sales"])
    return model


def train_xgboost(train, **kw):
    tr = _clean_train(train)
    model = xgb.XGBRegressor(**{**XGB_PARAMS, **kw})
    model.fit(tr[feature_columns(tr)], tr["sales"])
    return model


def predict(model, df):
    """Predict with exactly the columns the model was trained on, in the same
    order, and never below zero (you can't sell -3 units)."""
    cols = list(getattr(model, "feature_names_in_", feature_columns(df)))
    return np.clip(model.predict(df[cols]), 0, None)


# ---------------------------------------------------------------- prophet
def _prophet_one(g, horizon_dates, holidays_df):
    from prophet import Prophet
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, holidays=holidays_df, interval_width=0.8)
    m.fit(g.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]])
    fc = m.predict(pd.DataFrame({"ds": horizon_dates}))
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def prophet_forecast(train, horizon_dates, holidays_df=None, n_jobs=-1):
    """Fit one Prophet model per store-item series (in parallel)."""
    groups = list(train.groupby(["store", "item"]))
    outs = Parallel(n_jobs=n_jobs)(
        delayed(_prophet_one)(g[["date", "sales"]], horizon_dates, holidays_df) for _, g in groups)
    res = []
    for (key, _), fc in zip(groups, outs):
        fc = fc.rename(columns={"ds": "date"})
        fc["store"], fc["item"] = key
        res.append(fc)
    out = pd.concat(res, ignore_index=True)
    for c in ["yhat", "yhat_lower", "yhat_upper"]:
        out[c] = out[c].clip(lower=0)
    return out


# ---------------------------------------------------------------- CV
def time_series_cv(feat, n_folds=3, horizon=MAX_HORIZON, models=("naive", "ma", "lightgbm", "xgboost")):
    """Expanding-window cross-validation.

    Fold 1: train on data up to T-3h, test the next h days
    Fold 2: train up to T-2h, test the next h days ... and so on.
    This checks the model works in different seasons, not just one lucky window.
    """
    rows = []
    end = feat["date"].max()
    for k in range(n_folds, 0, -1):
        test_start = end - pd.Timedelta(days=k * horizon - 1)
        test_end = test_start + pd.Timedelta(days=horizon - 1)
        tr = feat[feat["date"] < test_start]
        te = feat[(feat["date"] >= test_start) & (feat["date"] <= test_end)]
        preds = {}
        if "naive" in models: preds["Naive (last year)"] = naive_last_year(te)
        if "ma" in models: preds["Moving avg (28d)"] = moving_average(tr, te)
        if "lightgbm" in models: preds["LightGBM"] = predict(train_lightgbm(tr), te)
        if "xgboost" in models: preds["XGBoost"] = predict(train_xgboost(tr), te)
        for name, p in preds.items():
            rows.append({"fold": f"{test_start.date()} → {test_end.date()}", "model": name,
                         **metrics(te["sales"], p)})
    return pd.DataFrame(rows)
