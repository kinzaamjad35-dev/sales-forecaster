"""
forecast.py - using trained models to look into the future, plus the
business tools built on top of a forecast:

  * future forecasts with an 80% prediction interval
  * hierarchical roll-ups (item -> store -> total)
  * what-if scenarios (promotion, price change)
  * inventory planning (safety stock, reorder point, order quantity)
  * anomaly detection on past sales
  * CSV / Excel export
"""
import io
import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import make_features, MAX_HORIZON
from .models import predict


# ---------------------------------------------------------------- forecasting
def forecast_future(history, point_model, lower_model=None, upper_model=None,
                    horizon=30):
    """Forecast `horizon` days after the last date in `history`.

    How it works: we add empty future rows to every series, compute features
    (they only use data >= 91 days old, so they're all known), then predict.
    """
    if horizon > MAX_HORIZON:
        raise ValueError(f"Horizon must be <= {MAX_HORIZON} days")
    last = history["date"].max()
    future_dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    keys = history[["store", "item"]].drop_duplicates()
    future = keys.merge(pd.DataFrame({"date": future_dates}), how="cross")
    future["sales"] = np.nan
    # Only the last ~2.1 years are needed to compute the longest lag (728 days).
    recent = history[history["date"] > last - pd.Timedelta(days=760)]
    full = make_features(pd.concat([recent, future], ignore_index=True))
    fut = full[full["date"] > last].copy()
    # Predict everything first, *then* add columns (a new column must not
    # be mistaken for a feature by the next model).
    point = predict(point_model, fut)
    if lower_model is not None and upper_model is not None:
        lo, hi = predict(lower_model, fut), predict(upper_model, fut)
    fut["forecast"] = point
    if lower_model is not None and upper_model is not None:
        # Quantile models are trained separately, so make sure lo <= point <= hi.
        fut["lower"] = np.minimum(lo, fut["forecast"])
        fut["upper"] = np.maximum(hi, fut["forecast"])
    return fut[["date", "store", "item", "forecast"] + (["lower", "upper"] if "lower" in fut else [])].reset_index(drop=True)


# ---------------------------------------------------------------- roll-ups
def hierarchical_rollup(fc, level="store"):
    """Bottom-up roll-up: add item forecasts to get store and total forecasts.
    level = "item" (no change), "store", or "total".
    Note: adding interval bounds is a conservative (wide) approximation."""
    cols = [c for c in ["forecast", "lower", "upper", "sales"] if c in fc]
    if level == "item":
        return fc
    by = ["date", "store"] if level == "store" else ["date"]
    return fc.groupby(by, as_index=False)[cols].sum()


# ---------------------------------------------------------------- what-if
def apply_scenario(fc, promo_uplift_pct=0.0, price_change_pct=0.0, elasticity=-1.5,
                   promo_days=None):
    """Adjust a forecast for a business scenario.

    promo_uplift_pct : e.g. 20 means a promotion adds +20% demand
    price_change_pct : e.g. -10 means a 10% price cut
    elasticity       : how sensitive buyers are to price. -1.5 means a 1% price
                       increase lowers demand by ~1.5%. (A common retail guess;
                       real values vary by product.)
    promo_days       : optional list of dates the promotion runs; default = all days
    """
    out = fc.copy()
    price_factor = (1 + price_change_pct / 100) ** elasticity
    promo = np.ones(len(out))
    if promo_uplift_pct:
        mask = out["date"].isin(pd.to_datetime(promo_days)) if promo_days is not None else np.ones(len(out), bool)
        promo = np.where(mask, 1 + promo_uplift_pct / 100, 1.0)
    factor = price_factor * promo
    for c in ["forecast", "lower", "upper"]:
        if c in out:
            out[c] = out[c] * factor
    return out


# ---------------------------------------------------------------- inventory
def inventory_plan(fc, current_stock, lead_time_days=7, service_level=0.95,
                   review_days=None, demand_std=None):
    """Classic inventory maths for one series.

    safety stock   = z * daily demand std * sqrt(lead time)
                     (z comes from the service level: 95% -> 1.645)
    reorder point  = average daily demand * lead time + safety stock
                     ("when stock falls to this level, order more")
    order quantity = forecast demand over the period + safety stock - current stock
    """
    review_days = review_days or len(fc)
    period = fc.head(review_days)
    daily_mean = period["forecast"].mean()
    if demand_std is None:
        # Estimate uncertainty from the 80% interval if we have it
        # (an 80% interval spans about 2 * 1.28 standard deviations).
        if "upper" in period:
            demand_std = ((period["upper"] - period["lower"]) / (2 * 1.2816)).mean()
        else:
            demand_std = period["forecast"].std()
    z = norm.ppf(service_level)
    safety = z * demand_std * np.sqrt(lead_time_days)
    rop = daily_mean * lead_time_days + safety
    demand = period["forecast"].sum()
    order = max(0.0, demand + safety - current_stock)
    days_cover = current_stock / daily_mean if daily_mean > 0 else np.inf
    return {
        "forecast_demand": demand, "avg_daily_demand": daily_mean,
        "daily_demand_std": demand_std, "z_score": z, "safety_stock": safety,
        "reorder_point": rop, "order_quantity": np.ceil(order),
        "days_of_cover": days_cover, "reorder_now": current_stock <= rop,
    }


# ---------------------------------------------------------------- anomalies
def detect_anomalies(series, window=28, threshold=3.5):
    """Flag unusual days in past sales.

    We compare each day with the *median* of the surrounding 4 weeks and
    measure the gap in robust units (MAD = median absolute deviation). A robust
    z-score above 3.5 is a common rule of thumb for "this is an outlier".
    `series` needs columns date and sales (one series).
    """
    s = series.sort_values("date").copy()
    med = s["sales"].rolling(window, center=True, min_periods=7).median()
    resid = s["sales"] - med
    mad = resid.abs().rolling(window, center=True, min_periods=7).median()
    s["expected"] = med
    s["robust_z"] = 0.6745 * resid / mad.replace(0, np.nan)
    s["is_anomaly"] = s["robust_z"].abs() > threshold
    return s


# ---------------------------------------------------------------- export
def to_excel_bytes(sheets: dict) -> bytes:
    """Write several DataFrames into one Excel file (one sheet each)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name[:31], index=False)
    return buf.getvalue()
