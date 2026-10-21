"""Inventory Planner: turns the forecast into a restocking decision using
safety stock, reorder point and a target service level."""
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import get_data_and_models, cached_forecast, data_key, series_picker
from src.forecast import inventory_plan, to_excel_bytes
import pandas as pd

st.set_page_config(page_title="Inventory Planner", page_icon="📦", layout="wide")
st.title("📦 Inventory Planner")
df, models = get_data_and_models()
store, item, horizon = series_picker(df, "inv_")
c = st.columns(3)
stock = c[0].number_input("Current stock (units)", 0, 100000, 300)
lead = c[1].number_input("Supplier lead time (days)", 1, 60, 7)
sl = c[2].slider("Service level (chance of NOT running out)", 0.80, 0.995, 0.95, 0.005)

fc = cached_forecast(df, models, data_key(), store, item, horizon)
plan = inventory_plan(fc, stock, lead_time_days=lead, service_level=sl)

m = st.columns(4)
m[0].metric("Forecast demand", f"{plan['forecast_demand']:,.0f}")
m[1].metric("Safety stock", f"{plan['safety_stock']:,.0f}")
m[2].metric("Reorder point", f"{plan['reorder_point']:,.0f}")
m[3].metric("Suggested order", f"{plan['order_quantity']:,.0f}")
if plan["reorder_now"]:
    st.error(f"⚠️ Stock ({stock}) is at or below the reorder point – order now. Covers ~{plan['days_of_cover']:.1f} days.")
else:
    st.success(f"✅ Stock covers ~{plan['days_of_cover']:.1f} days; above the reorder point.")

st.markdown(f"""
**How it's calculated**
- Safety stock = z ({plan['z_score']:.2f} for {sl:.1%}) × daily demand std ({plan['daily_demand_std']:.1f}) × √lead time ({lead}) = **{plan['safety_stock']:.0f}**
- Reorder point = avg daily demand ({plan['avg_daily_demand']:.1f}) × lead time + safety stock = **{plan['reorder_point']:.0f}**
- Order = forecast demand over {horizon} days + safety stock − current stock = **{plan['order_quantity']:.0f}**
""")

# Projected stock if we don't reorder
proj = fc[["date", "forecast"]].copy()
proj["stock_left"] = stock - proj["forecast"].cumsum()
fig = go.Figure([go.Scatter(x=proj["date"], y=proj["stock_left"], name="Projected stock (no reorder)")])
fig.add_hline(y=plan["reorder_point"], line_dash="dash", annotation_text="reorder point")
fig.add_hline(y=0, line_color="red")
st.plotly_chart(fig, width="stretch")

summary = pd.DataFrame([{"store": store, "item": item, "horizon_days": horizon, "current_stock": stock,
                         "lead_time": lead, "service_level": sl, **{k: round(float(v), 2) for k, v in plan.items()}}])
st.download_button("Download plan (Excel)", to_excel_bytes({"plan": summary, "forecast": fc.round({c: 2 for c in fc.select_dtypes('number')})}), "inventory_plan.xlsx")
