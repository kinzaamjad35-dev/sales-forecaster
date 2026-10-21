"""What-if scenarios: how would a promotion or a price change move demand?"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import get_data_and_models, cached_forecast, data_key, series_picker
from src.forecast import apply_scenario

st.set_page_config(page_title="What-If Scenarios", layout="wide")
st.title("What-If Scenarios")
st.caption("The model has no price/promotion data, so these are simple rule-based adjustments on top of the forecast.")
df, models = get_data_and_models()
store, item, horizon = series_picker(df, "wi_")
fc = cached_forecast(df, models, data_key(), store, item, horizon)

c = st.columns(4)
promo = c[0].slider("Promotion uplift (%)", 0, 100, 20)
promo_len = c[1].slider("Promotion length (days, from day 1)", 0, horizon, min(7, horizon))
price = c[2].slider("Price change (%)", -50, 50, 0)
elas = c[3].slider("Price elasticity", -4.0, 0.0, -1.5, 0.1,
                   help="-1.5: a 1% price rise lowers demand by about 1.5%")
unit_price = st.number_input("Current unit price (for revenue)", 0.0, 100000.0, 100.0)

promo_days = fc["date"].drop_duplicates().head(promo_len)
sc = apply_scenario(fc, promo, price, elas, promo_days=promo_days)

base_rev = fc["forecast"].sum() * unit_price
new_rev = sc["forecast"].sum() * unit_price * (1 + price / 100)
m = st.columns(3)
m[0].metric("Baseline demand", f"{fc['forecast'].sum():,.0f}")
m[1].metric("Scenario demand", f"{sc['forecast'].sum():,.0f}", f"{(sc['forecast'].sum()/fc['forecast'].sum()-1)*100:+.1f}%")
m[2].metric("Scenario revenue", f"{new_rev:,.0f}", f"{(new_rev/base_rev-1)*100:+.1f}%" if base_rev else None)

fig = go.Figure([go.Scatter(x=fc["date"], y=fc["forecast"], name="Baseline"),
                 go.Scatter(x=sc["date"], y=sc["forecast"], name="Scenario")])
st.plotly_chart(fig, width="stretch")
st.download_button("Download scenario CSV",
                   fc[["date", "forecast"]].rename(columns={"forecast": "baseline"})
                   .assign(scenario=sc["forecast"].values).round({"scenario": 2, "baseline": 2}).to_csv(index=False), "scenario.csv")
