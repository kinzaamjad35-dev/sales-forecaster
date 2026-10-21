"""Builds the README charts in docs/ from real data and real model output.
Run:  python tools/make_readme_assets.py   (~30 s; retrains LightGBM on the train split)"""
import os, sys, json
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.data import load_sales
from src.features import make_features, feature_columns
from src import models as M

NAVY, TEAL, AMBER, GREY, LIGHT = "#0F2C4C", "#0D9488", "#F59E0B", "#94A3B8", "#E2E8F0"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": LIGHT, "axes.labelcolor": NAVY,
    "xtick.color": NAVY, "ytick.color": NAVY, "axes.titlecolor": NAVY, "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": .8, "savefig.dpi": 130, "axes.axisbelow": True, "axes.ymargin": .12})
os.makedirs("docs", exist_ok=True)
stats = {}

# 1) Model comparison from the saved hold-out leaderboard
b = pd.read_csv("models/leaderboard_holdout.csv")
b = b[b.model.isin(["LightGBM", "XGBoost", "Prophet", "Naive (last year)"])].sort_values("MAE")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, col, lab in [(axes[0], "MAE", "MAE (units per day)"), (axes[1], "MAPE %", "MAPE (%)")]:
    cols = [TEAL if m == "LightGBM" else (GREY if m.startswith("Naive") else NAVY) for m in b.model]
    bars = ax.barh(b.model, b[col], color=cols); ax.invert_yaxis(); ax.set_title(lab); ax.grid(axis="y", visible=False)
    for r, v in zip(bars, b[col]): ax.text(v + b[col].max() * .01, r.get_y() + r.get_height() / 2, f"{v:.2f}", va="center", color=NAVY)
    ax.set_xlim(0, b[col].max() * 1.15)
fig.suptitle("Hold-out test, last 90 days of 2017, 500 series (lower is better)", color=NAVY, fontsize=11)
plt.tight_layout(); plt.savefig("docs/model_comparison.png"); plt.close()

# 2) Retrain LightGBM on the train split and predict the 90-day hold-out (same as the notebook)
df = load_sales("data/train.csv"); feat = make_features(df)
train, test, cutoff = M.time_split(feat, test_days=90)
model = M.train_lightgbm(train)
res = test[["date", "store", "item", "sales"]].assign(lgbm=M.predict(model, test), naive=M.naive_last_year(test))
stats["lgbm_mae"] = float((res.sales - res.lgbm).abs().mean()); stats["naive_mae"] = float((res.sales - res.naive).abs().mean())
tot = res.groupby("date")[["sales", "lgbm", "naive"]].sum()
hist = feat[(feat.date >= cutoff - pd.Timedelta(days=60)) & (feat.date < test.date.min())].groupby("date").sales.sum()

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(hist.index, hist.values, color=GREY, lw=1.4, label="History")
ax.plot(tot.index, tot.sales, color=NAVY, lw=1.8, label="Actual")
ax.plot(tot.index, tot.lgbm, color=TEAL, lw=1.8, label="LightGBM forecast")
ax.axvline(test.date.min(), color=AMBER, ls="--", lw=1); ax.text(test.date.min(), ax.get_ylim()[1], "  forecast starts", color=AMBER, va="top")
ax.set_title("Forecast vs actual: total daily sales, all 10 stores and 50 items"); ax.set_ylabel("units sold"); ax.legend(frameon=False, loc="lower left")
plt.tight_layout(); plt.savefig("docs/forecast_vs_actual.png"); plt.close()

# 3) Before / after on one series (store 1, item 1)
s = res[(res.store == 1) & (res.item == 1)]
mae_n, mae_l = (s.sales - s.naive).abs().mean(), (s.sales - s.lgbm).abs().mean()
stats["s1i1_naive_mae"], stats["s1i1_lgbm_mae"] = float(mae_n), float(mae_l)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, col, c, t, m in [(axes[0], "naive", GREY, "Before: copy last year's sales", mae_n), (axes[1], "lgbm", TEAL, "After: LightGBM forecast", mae_l)]:
    ax.plot(s.date, s.sales, color=NAVY, lw=1.3, label="Actual"); ax.plot(s.date, s[col], color=c, lw=1.8, label="Forecast")
    ax.set_title(f"{t}\nMAE {m:.2f} units/day"); ax.legend(frameon=False, loc="upper right"); ax.tick_params(axis="x", rotation=30)
fig.suptitle("Store 1, item 1, last 90 days of 2017", color=NAVY, fontsize=11)
plt.tight_layout(); plt.savefig("docs/before_after.png"); plt.close()

json.dump(stats, open("docs/readme_stats.json", "w"), indent=2); print(json.dumps(stats, indent=2))
