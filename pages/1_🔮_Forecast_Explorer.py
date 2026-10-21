"""Forecast Explorer: pick a store + item (or roll up to store / total level),
see the forecast with its 80% uncertainty band, and download it."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import get_data_and_models, cached_forecast, data_key
from src.forecast import hierarchical_rollup, to_excel_bytes

st.set_page_config(page_title="Forecast Explorer", page_icon="🔮", layout="wide")
st.title("🔮 Forecast Explorer")
df, models = get_data_and_models()

level = st.radio("Level", ["Item (one store)", "Store (all its items)", "Total (all stores)"], horizontal=True)
c1, c2, c3 = st.columns(3)
store = item = None
if level != "Total (all stores)":
    store = c1.selectbox("Store", sorted(df["store"].unique()))
if level.startswith("Item"):
    item = c2.selectbox("Item", sorted(df[df["store"] == store]["item"].unique()))
horizon = c3.slider("Horizon (days)", 7, 90, 30)
history_days = st.slider("Days of history to show", 30, 730, 180)

# Hierarchical roll-up: we always forecast at item level, then add up.
fc = cached_forecast(df, models, data_key(), store, item, horizon)
hist = df.copy()
if store is not None: hist = hist[hist["store"] == store]
if item is not None: hist = hist[hist["item"] == item]
roll = "item" if item is not None else ("store" if store is not None else "total")
fc_r = hierarchical_rollup(fc, roll)
hist_r = hist.groupby("date", as_index=False)["sales"].sum()
hist_r = hist_r[hist_r["date"] > hist_r["date"].max() - pd.Timedelta(days=history_days)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_r["date"], y=hist_r["sales"], name="Actual sales", line=dict(color="grey")))
fig.add_trace(go.Scatter(x=pd.concat([fc_r["date"], fc_r["date"][::-1]]),
                         y=pd.concat([fc_r["upper"], fc_r["lower"][::-1]]),
                         fill="toself", fillcolor="rgba(31,119,180,0.2)", line=dict(width=0),
                         name="80% prediction interval"))
fig.add_trace(go.Scatter(x=fc_r["date"], y=fc_r["forecast"], name="Forecast", line=dict(color="#1f77b4", width=3)))
fig.update_layout(title=f"{horizon}-day forecast", xaxis_title="date", yaxis_title="units")
st.plotly_chart(fig, width="stretch")

m = st.columns(3)
m[0].metric("Forecast total", f"{fc_r['forecast'].sum():,.0f} units")
m[1].metric("Avg per day", f"{fc_r['forecast'].mean():,.1f}")
same = hist_r.tail(horizon)["sales"].sum()
m[2].metric(f"vs last {horizon} days", f"{(fc_r['forecast'].sum()/same-1)*100:+.1f}%" if same else "n/a")

if roll == "store":
    st.subheader("Item breakdown")
    st.dataframe(fc.groupby("item", as_index=False)[["forecast", "lower", "upper"]].sum().round(0), width="stretch")

st.subheader("⬇️ Export")
out = fc_r.round({c: 2 for c in fc_r.select_dtypes('number')})
e1, e2 = st.columns(2)
e1.download_button("Download CSV", out.to_csv(index=False), "forecast.csv", "text/csv")
e2.download_button("Download Excel", to_excel_bytes({"forecast": out, "item_detail": fc.round({c: 2 for c in fc.select_dtypes('number')})}),
                   "forecast.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
st.dataframe(out, width="stretch")
