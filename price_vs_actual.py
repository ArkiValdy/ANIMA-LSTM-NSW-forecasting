"""
Direct comparisons of forecast prices against actual prices, built from the
archived walk-forward predictions and the live predicted-vs-actual table.

Outputs go to outputs/actual_comparison/:
  level_summary.csv     distribution of actual vs each forecast (mean, median, P95, max, bias)
  monthly_mean.csv      monthly mean actual price vs monthly mean forecast
  daily_mean.csv        daily average price: MAE, bias, correlation, within-10% share
  spike_detection.csv   hit rate / false alarms at the 130 A$/MWh spike threshold
  daily_peak.csv        daily maximum price: level error and timing of the peak
  live_days.csv         live forecast days, actual vs forecast level and peak
  fig_*.png             overlays and scatter
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import record_live
from aemo_benchmark import merge_aemo

OUT = config.OUTPUT_DIR / "actual_comparison"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"naive": "Seasonal naive", "arima": "ARIMA(0,1,5)",
          "hybrid": "Hybrid", "aemo": "AEMO pre-dispatch"}
COLORS = {"naive": "#9aa0a6", "arima": "#1f77b4", "hybrid": "#d62728", "aemo": "#2ca02c"}
# live_predicted.csv column for each model (AEMO is stored as aemo_pd there)
LIVE_COL = {"naive": "naive_pred", "arima": "arima_pred",
            "hybrid": "hybrid_pred", "aemo": "aemo_pd"}
THR = config.SPIKE_THRESHOLD


def load_backtest() -> pd.DataFrame:
    df = pd.read_csv(config.OUTPUT_DIR / "walkforward" / "predictions_price.csv",
                     index_col=0, parse_dates=True)
    return merge_aemo(df, "price").dropna(subset=["aemo_pred"])


def level_summary(df):
    rows = {"Actual": df["actual"]}
    rows.update({MODELS[m]: df[f"{m}_pred"] for m in MODELS})
    out = pd.DataFrame({k: {"mean": s.mean(), "median": s.median(),
                            "std": s.std(), "p95": s.quantile(0.95),
                            "max": s.max(), "min": s.min()}
                        for k, s in rows.items()}).T
    out["bias_vs_actual"] = out["mean"] - out.loc["Actual", "mean"]
    for m in MODELS:
        e = df[f"{m}_pred"] - df["actual"]
        out.loc[MODELS[m], "MAE_pct_of_mean_actual"] = 100 * e.abs().mean() / df["actual"].mean()
        out.loc[MODELS[m], "median_abs_err"] = e.abs().median()
        out.loc[MODELS[m], "share_within_20"] = 100 * (e.abs() <= 20).mean()
        out.loc[MODELS[m], "corr_with_actual"] = df[f"{m}_pred"].corr(df["actual"], method="spearman")
    return out


def monthly_mean(df):
    cols = ["actual"] + [f"{m}_pred" for m in MODELS]
    out = df[cols].resample("MS").mean()
    out.index = out.index.strftime("%Y-%m")
    return out.rename(columns={"actual": "Actual", **{f"{m}_pred": MODELS[m] for m in MODELS}})


def daily_mean(df):
    d = df.resample("D").mean(numeric_only=True)
    rows = {}
    for m in MODELS:
        e = d[f"{m}_pred"] - d["actual"]
        rows[MODELS[m]] = {
            "daily_mean_MAE": e.abs().mean(),
            "daily_mean_bias": e.mean(),
            "corr": d[f"{m}_pred"].corr(d["actual"], method="spearman"),
            "pct_days_within_10pct": 100 * (e.abs() <= 0.1 * d["actual"].abs()).mean(),
            "pct_days_within_20": 100 * (e.abs() <= 20).mean(),
        }
    return pd.DataFrame(rows).T, d


def spike_detection(df):
    act = df["actual"] > THR
    days_act = act.resample("D").max()
    rows = {}
    for m in MODELS:
        pred = df[f"{m}_pred"] > THR
        tp, fp, fn = (act & pred).sum(), (~act & pred).sum(), (act & ~pred).sum()
        # spike days: any interval > THR within the day
        days_pred = pred.resample("D").max()
        rows[MODELS[m]] = {
            "actual_spikes": int(act.sum()),
            "forecast_spikes": int(pred.sum()),
            "hits": int(tp), "misses": int(fn), "false_alarms": int(fp),
            "hit_rate_pct": 100 * tp / max(act.sum(), 1),
            "precision_pct": 100 * tp / max(pred.sum(), 1),
            "actual_spike_days": int(days_act.sum()),
            "spike_days_flagged": int((days_act & days_pred).sum()),
            "false_spike_days": int((~days_act & days_pred).sum()),
        }
    return pd.DataFrame(rows).T


def daily_peak(df):
    g = df.groupby(df.index.date)
    act_max = g["actual"].max()
    act_t = g["actual"].idxmax()
    rows = {}
    for m in MODELS:
        p_max = g[f"{m}_pred"].max()
        p_t = g[f"{m}_pred"].idxmax()
        hrs = ((act_t - p_t).dt.total_seconds().abs() / 3600).to_numpy()
        rows[MODELS[m]] = {
            "mean_actual_daily_max": act_max.mean(),
            "mean_forecast_daily_max": p_max.mean(),
            "median_abs_err_daily_max": (p_max - act_max).abs().median(),
            "pct_peak_within_1h": 100 * (hrs <= 1).mean(),
            "median_peak_timing_err_h": np.median(hrs),
        }
    return pd.DataFrame(rows).T


def live_days():
    lv = pd.read_csv(record_live.TABLE_CSV, parse_dates=["timestamp"])
    lv = lv.dropna(subset=["actual_price"])
    lv = lv[lv["timestamp"].dt.date.astype(str) == lv["forecast_date"]]
    rows = []
    for day, g in lv.groupby("forecast_date"):
        g = g.set_index("timestamp")
        r = {"date": day, "issued_at": g["issued_at"].iloc[0], "n": len(g),
             "actual_mean": g["actual_price"].mean(),
             "actual_max": g["actual_price"].max(),
             "actual_max_time": g["actual_price"].idxmax().strftime("%H:%M")}
        for m, col in LIVE_COL.items():
            r[f"{m}_mean"] = g[col].mean()
            r[f"{m}_max"] = g[col].max()
            r[f"{m}_max_time"] = g[col].idxmax().strftime("%H:%M")
            r[f"{m}_MAE"] = (g[col] - g["actual_price"]).abs().mean()
            r[f"{m}_median_AE"] = (g[col] - g["actual_price"]).abs().median()
        rows.append(r)
    return pd.DataFrame(rows), lv


def plot_overlay(df, start, end, fname, title):
    w = df.loc[start:end]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(w.index, w["actual"], color="black", lw=1.6, label="Actual")
    for m, c in COLORS.items():
        ax.plot(w.index, w[f"{m}_pred"], lw=1, color=c, label=MODELS[m])
    ax.set_ylabel("A$/MWh")
    ax.set_title(title)
    ax.legend(ncol=5, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def plot_daily_mean(d):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d.index, d["actual"], color="black", lw=1.6, label="Actual")
    for m, c in COLORS.items():
        ax.plot(d.index, d[f"{m}_pred"], lw=1, color=c, label=MODELS[m])
    ax.set_yscale("symlog", linthresh=200)
    ax.set_ylabel("Daily mean price, A$/MWh (symlog)")
    ax.set_title("Daily mean price: actual vs forecast, 1 Jan to 30 Jun 2026")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_daily_mean.png", dpi=150)
    plt.close(fig)


def plot_scatter(df):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharex=True, sharey=True)
    lim = (-50, 400)
    for ax, m in zip(axes, MODELS):
        ax.scatter(df["actual"], df[f"{m}_pred"], s=2, alpha=0.25)
        ax.plot(lim, lim, color="black", lw=0.8)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(MODELS[m], fontsize=10)
        ax.set_xlabel("Actual (A$/MWh)")
    axes[0].set_ylabel("Forecast (A$/MWh)")
    fig.suptitle("Forecast vs actual price, intervals between -50 and 400 A$/MWh")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scatter.png", dpi=150)
    plt.close(fig)


def plot_live(lv):
    for day, g in lv.groupby("forecast_date"):
        g = g.set_index("timestamp")
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(g.index, g["actual_price"], color="black", lw=1.8, label="Actual")
        for m, col in LIVE_COL.items():
            # live legend says plain "ARIMA", not the backtest's "ARIMA(0,1,5)"
            lab = "ARIMA" if m == "arima" else MODELS[m]
            ax.plot(g.index, g[col], lw=1, color=COLORS[m], label=lab)
        ax.set_yscale("symlog", linthresh=500)
        ax.set_ylabel("A$/MWh (symlog)")
        ax.set_title(f"Live forecast vs actual, {day}")
        ax.legend(ncol=5, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_live_{day}.png", dpi=150)
        plt.close(fig)


def main():
    df = load_backtest()
    pd.set_option("display.width", 200, "display.float_format", "{:.1f}".format)
    tables = {"level_summary": level_summary(df), "monthly_mean": monthly_mean(df)}
    tables["daily_mean"], d = daily_mean(df)
    tables["spike_detection"] = spike_detection(df)
    tables["daily_peak"] = daily_peak(df)
    tables["live_days"], lv = live_days()
    for name, t in tables.items():
        t.to_csv(OUT / f"{name}.csv")
        print(f"\n== {name} ==\n{t.T if name == 'live_days' else t}")

    plot_overlay(df, "2026-04-06", "2026-04-12 23:30", "fig_overlay_calm_week.png",
                 "Forecast vs actual price, calm week (6 to 12 April 2026)")
    plot_overlay(df, "2026-02-02", "2026-02-08 23:30", "fig_overlay_spike_week.png",
                 "Forecast vs actual price, spike week (2 to 8 February 2026)")
    plot_daily_mean(d)
    plot_scatter(df)
    plot_live(lv)


if __name__ == "__main__":
    main()
