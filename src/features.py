"""
features.py - turning raw sales into "clues" (features) a model can learn from.

A tree model like XGBoost can't look at a calendar by itself, so we give it
columns such as "day of week", "sales 91 days ago", "average of the last 4
weeks (as of 91 days ago)" and "is it a holiday?".

WHY ALL LAGS ARE >= 91 DAYS:
We want to forecast up to 90 days ahead. On the day we make the forecast we
only know sales up to *today*. If the model used "yesterday's sales" it would
need values we don't have yet when predicting 60 days out. By using only
information at least 91 days old, every feature is available for the whole
forecast window. This is called "direct" forecasting and it prevents leakage.
"""
import numpy as np
import pandas as pd
import holidays
from hijridate import Hijri, Gregorian   # converts Islamic (Hijri) dates to normal dates

# ---------------------------------------------------------------- settings
# Which country's holidays to use. "PK" = Pakistan (default): public holidays
# PLUS Ramadan and both Eids. Change to "US" to go back to the US calendar
# (then retrain: run the notebook). The app and notebook both read this.
COUNTRY = "PK"
EID_WINDOW = 7            # how many days before/after Eid we track

MAX_HORIZON = 90          # longest forecast the app allows (days)
LAGS = [91, 98, 105, 112, 119, 126, 182, 364, 371, 546, 728]
ROLL_WINDOWS = [7, 28, 91, 182, 364]


def holiday_calendar(start, end, country=COUNTRY):
    """Set of public-holiday dates for `country` ("PK" or "US").
    The `holidays` library already knows Pakistan's Eid dates (it calculates
    future years too, marked "estimated")."""
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 2)
    return pd.to_datetime(list(holidays.country_holidays(country, years=years).keys()))


def islamic_dates(year_from=2012, year_to=2031):
    """Ramadan and Eid dates (Pakistan) for every year in the range.

    Returns a table with one row per Islamic year: ramadan_start, eid_fitr,
    eid_adha. Eid dates come from the `holidays` library's Pakistan calendar
    (first day of each Eid). Ramadan's start comes from `hijridate`
    (Umm al-Qura calendar), shifted by the same 0-1 day gap that Pakistan's
    moon sighting usually has, so Ramadan always ends the day before Eid.
    These are calendar facts known years ahead, so they can't leak the future.
    """
    pk = holidays.country_holidays("PK", years=range(year_from, year_to + 1))

    def first_days(word):
        # Each Eid is a 3-day holiday; keep only the first day of each.
        days = {x for x, n in pk.items() if word in n}
        return sorted(x for x in days if x - pd.Timedelta(days=1).to_pytimedelta() not in days)

    fitr, adha = first_days("Fitr"), first_days("Adha")
    rows = []
    for e in fitr:
        hy = Gregorian(e.year, e.month, e.day).to_hijri().year   # Islamic year
        uq_eid = pd.Timestamp(Hijri(hy, 10, 1).to_gregorian().isoformat())
        shift = pd.Timestamp(e) - uq_eid          # Pakistan vs Saudi calendar gap
        start = pd.Timestamp(Hijri(hy, 9, 1).to_gregorian().isoformat()) + shift
        adha_same = [a for a in adha if 0 < (pd.Timestamp(a) - pd.Timestamp(e)).days < 80]
        rows.append({"ramadan_start": start, "eid_fitr": pd.Timestamp(e),
                     "eid_adha": pd.Timestamp(adha_same[0]) if adha_same else pd.NaT})
    return pd.DataFrame(rows)


def _days_to_nearest(d, events, sign):
    """For each date: days until (sign=+1) or since (sign=-1) the closest
    event, if it's within EID_WINDOW days; otherwise -1 (meaning "not near")."""
    ev = np.array(sorted(pd.to_datetime(events).dropna()), dtype="datetime64[D]")
    days = d.values.astype("datetime64[D]")
    out = np.full(len(days), -1, dtype="int8")
    for e in ev:
        gap = ((e - days) * sign).astype(int)       # positive = before/after
        hit = (gap >= 1) & (gap <= EID_WINDOW) & ((out == -1) | (gap < out))
        out[hit] = gap[hit]
    return out


def add_islamic_features(df):
    """Ramadan + Eid columns (used when COUNTRY is "PK").
    - is_ramadan: 1 during the fasting month
    - ramadan_day: 1..30 inside Ramadan, 0 otherwise
    - is_eid_fitr / is_eid_adha: 1 on the Eid holidays themselves
    - days_to_eid_*: 1..7 in the week before Eid (shopping rush), else -1
    - days_after_eid_*: 1..7 in the week after Eid, else -1"""
    d = df["date"]
    isl = islamic_dates(d.min().year - 1, d.max().year + 1)
    df["ramadan_day"] = 0
    for _, r in isl.iterrows():
        inside = (d >= r.ramadan_start) & (d < r.eid_fitr)
        df.loc[inside, "ramadan_day"] = (d[inside] - r.ramadan_start).dt.days + 1
    df["ramadan_day"] = df["ramadan_day"].astype("int8")
    df["is_ramadan"] = (df["ramadan_day"] > 0).astype("int8")
    pk = holidays.country_holidays("PK", years=range(d.min().year, d.max().year + 2))
    for key, word in [("fitr", "Fitr"), ("adha", "Adha")]:
        days = pd.to_datetime([x for x, n in pk.items() if word in n])
        df[f"is_eid_{key}"] = d.isin(days).astype("int8")
        df[f"days_to_eid_{key}"] = _days_to_nearest(d, isl[f"eid_{key}"], +1)
        df[f"days_after_eid_{key}"] = _days_to_nearest(d, isl[f"eid_{key}"], -1)
    return df


def add_calendar_features(df, country=COUNTRY):
    """Calendar columns: day of week, month, etc. plus holiday flags."""
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
    hol = holiday_calendar(d.min(), d.max(), country)
    df["is_holiday"] = d.isin(hol).astype("int8")
    # Shops are often busy the day *before* a holiday.
    df["is_pre_holiday"] = (d + pd.Timedelta(days=1)).isin(hol).astype("int8")
    if country == "PK":
        df = add_islamic_features(df)
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


def make_features(df, country=COUNTRY):
    """Main entry point: returns the data with all feature columns added."""
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    df = add_calendar_features(df, country)
    df = add_lag_features(df)
    return df


def feature_columns(df):
    """Every column except the date and the answer (sales).
    Store and item numbers are kept so one "global" model can learn that
    e.g. item 15 always sells more than item 5."""
    return [c for c in df.columns if c not in ("date", "sales")]
