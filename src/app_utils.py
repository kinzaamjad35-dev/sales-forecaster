"""
app_utils.py - helpers shared by every page of the Streamlit dashboard.

Streamlit re-runs the whole script on every click, so we use caching
(@st.cache_data / @st.cache_resource) to avoid reloading data or retraining
models each time.
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from .data import load_sales, clean_sales, SAMPLE_PATH
from .features import make_features
from . import models as M
from .forecast import forecast_future

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "lgbm_models.joblib"


@st.cache_data(show_spinner="Loading sample data…")
def sample_data():
    return load_sales(SAMPLE_PATH)


@st.cache_data(show_spinner="Reading your CSV…")
def uploaded_data(file_bytes: bytes):
    import io
    return load_sales(io.BytesIO(file_bytes))


@st.cache_resource(show_spinner="Training models on your data (about a minute)…")
def train_models(_df, key: str):
    """Train point + 10%/90% quantile LightGBM models. `key` tells the cache
    which dataset this is (the underscore stops Streamlit hashing a big table)."""
    feat = make_features(_df)
    return {"point": M.train_lightgbm(feat),
            "lower": M.train_lightgbm(feat, objective="quantile", alpha=0.1),
            "upper": M.train_lightgbm(feat, objective="quantile", alpha=0.9)}


@st.cache_resource(show_spinner="Loading trained models…")
def pretrained_models():
    return joblib.load(MODEL_PATH)


def sidebar_data_picker():
    """Shown on every page: choose sample data or upload a CSV."""
    with st.sidebar:
        st.header("Data")
        up = st.file_uploader("Upload CSV (date, item, sales [, store])", type="csv")
        if up is not None:
            st.session_state["upload"] = up.getvalue()
            st.session_state["upload_name"] = up.name
        if "upload" in st.session_state:
            st.caption(f"Using **{st.session_state['upload_name']}**")
            if st.button("Use sample data instead"):
                del st.session_state["upload"]
                st.rerun()
        else:
            st.caption("Using Kaggle sample: 10 stores × 50 items, 2013–2017")


def get_data_and_models():
    sidebar_data_picker()
    if "upload" in st.session_state:
        try:
            df = uploaded_data(st.session_state["upload"])
            if df["date"].nunique() < 400:
                st.warning("Less than ~400 days of history: the yearly features will be mostly empty, so forecasts will be rough.")
            return df, train_models(df, key=str(hash(st.session_state["upload"])))
        except (ValueError, pd.errors.ParserError) as e:
            st.error(f"Could not use **{st.session_state['upload_name']}**: {e}")
            st.stop()
    df = sample_data()
    if MODEL_PATH.exists():
        return df, pretrained_models()
    return df, train_models(df, key="sample")


@st.cache_data(show_spinner="Forecasting…")
def cached_forecast(_df, _models, key, store, item, horizon):
    """Forecast one store-item series (or all, if store/item is None)."""
    h = _df
    if store is not None: h = h[h["store"] == store]
    if item is not None: h = h[h["item"] == item]
    return forecast_future(h, _models["point"], _models["lower"], _models["upper"], horizon=horizon)


def data_key():
    return str(hash(st.session_state["upload"])) if "upload" in st.session_state else "sample"


def series_picker(df, prefix=""):
    c1, c2, c3 = st.columns(3)
    store = c1.selectbox("Store", sorted(df["store"].unique()), key=prefix + "store")
    item = c2.selectbox("Item", sorted(df[df["store"] == store]["item"].unique()), key=prefix + "item")
    horizon = c3.slider("Horizon (days)", 7, 90, 30, key=prefix + "h")
    return store, item, horizon
