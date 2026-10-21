"""Anomaly detection: find past days where sales were unusually high or low."""
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import get_data_and_models
from src.forecast import detect_anomalies

st.set_page_config(page_title="Anomalies", layout="wide")
st.title("Anomaly Detection")
df, _ = get_data_and_models()
c = st.columns(3)
store = c[0].selectbox("Store", sorted(df["store"].unique()))
item = c[1].selectbox("Item", sorted(df[df["store"] == store]["item"].unique()))
thr = c[2].slider("Sensitivity (robust z threshold)", 2.0, 6.0, 3.5, 0.5, help="Lower = more days flagged")

an = detect_anomalies(df[(df["store"] == store) & (df["item"] == item)][["date", "sales"]], threshold=thr)
flag = an[an["is_anomaly"]]
st.metric("Anomalous days", len(flag))
fig = go.Figure([go.Scatter(x=an["date"], y=an["sales"], name="sales", line=dict(color="lightgrey")),
                 go.Scatter(x=an["date"], y=an["expected"], name="expected (rolling median)"),
                 go.Scatter(x=flag["date"], y=flag["sales"], mode="markers", name="anomaly",
                            marker=dict(color="red", size=9))])
st.plotly_chart(fig, width="stretch")
st.dataframe(flag[["date", "sales", "expected", "robust_z"]].round({"sales": 2, "expected": 2, "robust_z": 2}), width="stretch", hide_index=True)
