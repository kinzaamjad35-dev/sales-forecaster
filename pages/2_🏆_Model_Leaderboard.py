"""Model Leaderboard: results saved by the notebook (baselines vs Prophet vs
XGBoost vs LightGBM), cross-validation, interval coverage and SHAP."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Model Leaderboard", page_icon="🏆", layout="wide")
st.title("🏆 Model Leaderboard")
MD = Path(__file__).resolve().parent.parent / "models"
IMG = MD.parent / "images"

if not (MD / "leaderboard_holdout.csv").exists():
    st.warning("Run notebooks/01_explore_and_model.ipynb first to create the results.")
    st.stop()

st.caption("Sample data · test = last 90 days of 2017 · lower is better. MAE = units off per day.")
board = pd.read_csv(MD / "leaderboard_holdout.csv")
st.subheader("Hold-out test (Oct–Dec 2017)")
st.dataframe(board.round(3), width="stretch", hide_index=True)
st.plotly_chart(px.bar(board, x="model", y="MAE", color="model", title="MAE by model"), width="stretch")

st.subheader("Time-series cross-validation (3 folds × 90 days)")
st.caption("Prophet isn't in CV because 500 fits × 3 folds is slow; it's compared on the hold-out above.")
cv = pd.read_csv(MD / "cv_folds.csv")
st.plotly_chart(px.bar(cv, x="fold", y="MAE", color="model", barmode="group"), width="stretch")
st.dataframe(pd.read_csv(MD / "cv_summary.csv"), width="stretch", hide_index=True)

if (MD / "interval_coverage.csv").exists():
    cov = pd.read_csv(MD / "interval_coverage.csv", index_col=0).iloc[:, 0]
    st.subheader("Prediction intervals")
    c = st.columns(2)
    c[0].metric("LightGBM quantile 80% interval coverage", f"{cov['lightgbm_quantile_coverage_pct']:.1f}%")
    c[1].metric("Prophet 80% interval coverage", f"{cov['prophet_coverage_pct']:.1f}%")
    st.caption("Ideal = 80%: that many actual days should fall inside the band.")

st.subheader("🧠 SHAP: what drives the LightGBM forecast?")
imp = pd.read_csv(MD / "shap_importance.csv").head(15)
st.plotly_chart(px.bar(imp[::-1], x="mean_abs_shap", y="feature", orientation="h"), width="stretch")
if (IMG / "shap_summary.png").exists():
    st.image(str(IMG / "shap_summary.png"))
