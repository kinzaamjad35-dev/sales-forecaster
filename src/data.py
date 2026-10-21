"""
data.py - loading and cleaning sales data.

Every other part of the project (notebook, app) calls these functions, so the
data is always prepared the same way.

Expected columns: date, item, sales  (and optionally store).
If there is no "store" column we assume a single shop (store = 1).
"""
from pathlib import Path
import pandas as pd

# Path to the real Kaggle "Store Item Demand Forecasting" training file.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_PATH = DATA_DIR / "train.csv"


def load_sales(source=SAMPLE_PATH) -> pd.DataFrame:
    """Read a CSV (a file path or an uploaded file object) and clean it."""
    df = pd.read_csv(source)
    return clean_sales(df)


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Make the data tidy so the models can trust it."""
    df = df.copy()
    # Column names in lowercase without spaces, so "Sales " becomes "sales".
    df.columns = [c.strip().lower() for c in df.columns]

    missing = {"date", "item", "sales"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {sorted(missing)}")

    if "store" not in df.columns:
        df["store"] = 1  # one-shop data

    # Turn text dates into real dates; rows with unreadable dates are dropped.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "sales"])

    # Sales can't be negative (returns are ignored in this simple project).
    df["sales"] = df["sales"].clip(lower=0)

    # If the same store/item/day appears twice, add them together.
    df = df.groupby(["date", "store", "item"], as_index=False)["sales"].sum()

    # Fill missing days with 0 sales so every series has one row per day.
    # (A missing day usually means nothing was sold.)
    full = []
    for (s, i), g in df.groupby(["store", "item"]):
        days = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(days).rename_axis("date").reset_index()
        g["store"], g["item"] = s, i
        g["sales"] = g["sales"].fillna(0)
        full.append(g)
    df = pd.concat(full, ignore_index=True)

    # Small number types save memory (the sample has ~913,000 rows).
    df["sales"] = df["sales"].astype("float32")
    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)
