"""
data.py - loading and cleaning sales data.

Every other part of the project (notebook, app) calls these functions, so the
data is always prepared the same way.

Expected columns: date, item, sales  (and optionally store).
If there is no "store" column we assume a single shop (store = 1).
"""
from pathlib import Path
import numpy as np
import pandas as pd

# Path to the real Kaggle "Store Item Demand Forecasting" training file.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_PATH = DATA_DIR / "train.csv"


def load_sales(source=SAMPLE_PATH) -> pd.DataFrame:
    """Read a CSV (a file path or an uploaded file object) and clean it."""
    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.") from None
    return clean_sales(df)


# Common alternative column names, mapped to the names the models expect.
COLUMN_ALIASES = {
    "date": ["day", "order_date", "sale_date", "sales_date", "transaction_date", "ds"],
    "item": ["product", "product_id", "item_id", "sku", "item_name", "product_name", "article"],
    "store": ["shop", "branch", "store_id", "outlet", "location"],
    "sales": ["qty", "quantity", "units", "units_sold", "quantity_sold", "sold", "demand", "y"],
}


def _rename_aliases(df):
    cols = set(df.columns)
    ren = {}
    for target, alts in COLUMN_ALIASES.items():
        if target in cols:
            continue
        for a in alts:
            if a in cols and a not in ren:
                ren[a] = target
                break
    return df.rename(columns=ren)


def _parse_dates(col):
    """Parse dates, choosing day-first (31/12/2024, common in Pakistan) or
    month-first (12/31/2024) by whichever reads more rows; on a tie, the one
    giving the shorter overall date range (wrongly swapped day/month scatters dates)."""
    col = col.astype(str).str.strip()
    if col.str.match(r"^\d{4}-\d{1,2}-\d{1,2}").mean() > 0.9:  # ISO: unambiguous
        return pd.to_datetime(col, errors="coerce", format="mixed")
    cands = [pd.to_datetime(col, errors="coerce", format="mixed", dayfirst=f) for f in (False, True)]
    def key(d):
        span = (d.max() - d.min()) if d.notna().any() else pd.Timedelta(0)
        return (d.isna().sum(), span)
    return min(cands, key=key)


def _to_number(col):
    """Numbers like "1,234", "Rs 1,200" or " 15 " -> float; blanks -> NaN."""
    if pd.api.types.is_numeric_dtype(col):
        return col.astype(float)
    cleaned = col.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned.replace("", None), errors="coerce")


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Make the data tidy so the models can trust it."""
    df = df.copy()
    # Column names in lowercase without spaces, so "Sales " becomes "sales".
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = _rename_aliases(df)

    missing = {"date", "item", "sales"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {sorted(missing)}")

    if "store" not in df.columns:
        df["store"] = 1  # one-shop data

    # Turn text dates into real dates; rows with unreadable dates are dropped.
    df["date"] = _parse_dates(df["date"])
    df["sales"] = _to_number(df["sales"])
    df = df.dropna(subset=["date"])
    if df.empty or df["sales"].notna().sum() == 0:
        raise ValueError("No usable rows: check the date and sales columns contain dates and numbers.")

    # The models need numeric store/item ids. Text ids (e.g. "Shirt") are
    # numbered 1, 2, 3 ... in alphabetical order.
    for c in ("store", "item"):
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
            codes = {v: i + 1 for i, v in enumerate(sorted(df[c].unique()))}
            df[c] = df[c].map(codes)

    # Sales can't be negative (returns are ignored in this simple project).
    df["sales"] = df["sales"].clip(lower=0)

    # If the same store/item/day appears twice, add them together.
    # (min_count=1 keeps a day whose only value is blank as NaN, not 0.)
    df = df.groupby(["date", "store", "item"], as_index=False)["sales"].sum(min_count=1)

    # Fill missing days with 0 sales so every series has one row per day.
    # (A missing day usually means nothing was sold.)
    full = []
    for (s, i), g in df.groupby(["store", "item"]):
        days = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date")
        blank = g.index[g["sales"].isna()]       # row present, value missing
        g = g.reindex(days).rename_axis("date")
        g["sales"] = g["sales"].fillna(0)        # day absent = nothing sold
        # A blank cell is unknown, not zero: fill it from neighbouring days.
        g.loc[blank, "sales"] = np.nan
        g["sales"] = g["sales"].interpolate(limit_direction="both").fillna(0)
        g = g.reset_index()
        g["store"], g["item"] = s, i
        full.append(g)
    df = pd.concat(full, ignore_index=True)

    # Small number types save memory (the sample has ~913,000 rows).
    df["sales"] = df["sales"].astype("float32")
    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)
