"""Post hoc diagnostic corrections for Section 4.6 of the final report.
 
Tests two corrections on the stored walk forward predictions
(outputs/walkforward/predictions_price.csv), both using only
information available at issue time:
  1. Cap the seasonal naive base before adding the LSTM residual.
  2. Subtract a trailing mean of the hybrid's daily residuals.
 
Run:  python posthoc_corrections.py path/to/predictions_price.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
 
path = sys.argv[1] if len(sys.argv) > 1 else "outputs/walkforward/predictions_price.csv"
bt = pd.read_csv(path, parse_dates=["timestamp"])
bt["date"] = bt["timestamp"].dt.date
sp = bt["is_spike"] == 1
 
 
def seg_scores(pred, name):
    row = {"model": name}
    for mask, lab in ((slice(None), "overall"), (~sp, "normal"), (sp, "spike")):
        e = (bt["actual"] - pred)[mask]
        row[f"{lab}_rmse"] = round(float(np.sqrt((e ** 2).mean())), 1)
        row[f"{lab}_mae"] = round(float(e.abs().mean()), 1)
    return row
 
 
def daily_rmse(pred):
    return (bt["actual"] - pred).pow(2).groupby(bt["date"]).mean().pow(0.5)
 
 
rows = [seg_scores(bt["naive_pred"], "naive"),
        seg_scores(bt["arima_pred"], "arima"),
        seg_scores(bt["hybrid_pred"], "hybrid")]
 
# Correction 1: cap the naive base (sensitivity over thresholds)
for cap in (150, 200, 300, 500, 1000):
    capped = np.minimum(bt["naive_pred"], float(cap))
    rows.append(seg_scores(capped, f"naive_cap{cap}"))
    rows.append(seg_scores(capped + bt["lstm_resid"], f"hybrid_cap{cap}"))
 
# Correction 2: trailing bias correction on the hybrid
bt["hyb_resid"] = bt["actual"] - bt["hybrid_pred"]
daily_resid = bt.groupby("date")["hyb_resid"].mean()
for k in (7, 14, 30):
    corr = daily_resid.shift(1).rolling(k, min_periods=3).mean()  # past days only
    c = bt["date"].map(corr).fillna(0.0)
    rows.append(seg_scores(bt["hybrid_pred"] + c, f"hybrid_bias{k}d"))
 
out = pd.DataFrame(rows)
print(out.to_string(index=False))
 
# Paired t-tests on daily RMSE (n = number of forecast days)
cap300_naive = np.minimum(bt["naive_pred"], 300.0)
cap300_hybrid = cap300_naive + bt["lstm_resid"]
pairs = [("naive vs naive_cap300", bt["naive_pred"], cap300_naive),
         ("arima vs naive_cap300", bt["arima_pred"], cap300_naive),
         ("hybrid vs hybrid_cap300", bt["hybrid_pred"], cap300_hybrid)]
print()
for name, a, b in pairs:
    t, p = stats.ttest_rel(daily_rmse(a), daily_rmse(b))
    print(f"{name:26s} t = {t:5.2f}  p = {p:.4g}  (n = {bt['date'].nunique()})")