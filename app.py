"""
app.py - the home page (Overview) of the Streamlit dashboard.

Run with:   streamlit run app.py
Other pages live in the pages/ folder; Streamlit adds them to the sidebar
automatically (the number at the start of each file name sets the order).
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_utils import get_data_and_models

st.set_page_config(page_title="AI Sales Forecaster", layout="wide")
st.title("AI Sales Forecaster")
st.caption("Forecast demand, compare models, and plan inventory. Pages in the sidebar →")

df, models = get_data_and_models()

# ---- KPI cards ------------------------------------------------------------
last = df["date"].max()
last30 = df[df["date"] > last - pd.Timedelta(days=30)]["sales"].sum()
prev30 = df[(df["date"] <= last - pd.Timedelta(days=30)) &
            (df["date"] > last - pd.Timedelta(days=60))]["sales"].sum()
yoy = df[df["date"].dt.year == last.year]["sales"].sum()
yoy_prev = df[df["date"].dt.year == last.year - 1]["sales"].sum()

k = st.columns(5)
k[0].metric("Total units sold", f"{df['sales'].sum():,.0f}")
k[1].metric("Stores · Items", f"{df['store'].nunique()} · {df['item'].nunique()}")
k[2].metric("Date range", f"{df['date'].min():%b %Y} – {last:%b %Y}")
k[3].metric("Last 30 days", f"{last30:,.0f}", f"{(last30/prev30-1)*100:+.1f}% vs previous 30" if prev30 else None)
k[4].metric(f"{last.year} total", f"{yoy:,.0f}", f"{(yoy/yoy_prev-1)*100:+.1f}% YoY" if yoy_prev else None)

# ---- Charts ---------------------------------------------------------------
monthly = df.set_index("date")["sales"].resample("MS").sum().reset_index()
st.plotly_chart(px.line(monthly, x="date", y="sales", title="Monthly total sales"), width="stretch")

c1, c2 = st.columns(2)
by_store = df.groupby("store", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
c1.plotly_chart(px.bar(by_store, x="store", y="sales", title="Sales by store"), width="stretch")
top = df.groupby("item", as_index=False)["sales"].sum().nlargest(10, "sales")
c2.plotly_chart(px.bar(top, x="item", y="sales", title="Top 10 items"), width="stretch")

wd = df.groupby(df["date"].dt.day_name())["sales"].mean().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reset_index()
st.plotly_chart(px.bar(wd, x="date", y="sales", title="Average daily sales by weekday",
                       labels={"date": "weekday"}), width="stretch")
