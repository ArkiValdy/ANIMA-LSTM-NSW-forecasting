"""
Report figures 2-11 (figure 1, the pipeline diagram, lives in the report doc).

Everything is read from archived files, so no model is re-run:
  outputs/walkforward/predictions_price.csv   Jan-Jun 2026 backtest (naive, ARIMA, hybrid)
  data/raw/predispatch_price_30min.csv        AEMO pre-dispatch for the backtest
  outputs/live_predicted.csv                  live day-ahead track (Oct 2026)
  data/raw/price_demand_30min_live.csv        actual prices (for 12 Oct)

Run:  .venv/Scripts/python.exe make_report_figures.py
Out:  outputs/report_figures/fig02_*.png ... fig11_*.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

import config
import record_live
from aemo_benchmark import merge_aemo

OUT = config.OUTPUT_DIR / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

# Colour follows the model, the same in every figure (validated categorical palette).
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
C = {"hybrid": "#2a78d6", "naive": "#eb6834", "arima": "#1baf7a", "aemo": "#eda100", "actual": INK}
LABEL = {"hybrid": "Hybrid", "naive": "Seasonal naive", "arima": "ARIMA",
         "aemo": "AEMO pre-dispatch", "actual": "Actual"}
MODELS = ["naive", "arima", "hybrid", "aemo"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "font.size": 9.5, "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlepad": 18, "axes.titlelocation": "left", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "xtick.color": INK2, "ytick.color": INK2,
    "legend.frameon": False, "lines.linewidth": 1.6, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "lines.solid_capstyle": "round",
})
PRICE = "Price (A$/MWh)"


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("  wrote", name)


def symlog(ax, linthresh=300):
    """Linear up to `linthresh`, log above, so normal prices and spikes share one frame."""
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.6)
    ticks = [-100, 0, 100, 200, 300, 1000, 3000, 10000, 20000]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:,}" for t in ticks])
    ax.axhline(linthresh, color=MUTED, lw=0.6, ls=(0, (2, 3)), zorder=0)


# ------------------------------------------------------------------ data
def load_backtest() -> pd.DataFrame:
    df = pd.read_csv(config.OUTPUT_DIR / "walkforward" / "predictions_price.csv",
                     index_col=0, parse_dates=True)
    return merge_aemo(df, "price")


def load_live() -> pd.DataFrame:
    df = pd.read_csv(record_live.TABLE_CSV, parse_dates=["timestamp"])
    return df.rename(columns={"aemo_pd": "aemo_pred"}).set_index("timestamp")


# ------------------------------------------------------------------ fig 2
def fig02_timeline(live: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.grid(False)
    ax.spines[["left"]].set_visible(False)
    start, end = pd.Timestamp("2025-12-01"), pd.Timestamp("2026-11-01")

    # southern-hemisphere seasons as a quiet background band
    seasons = [("Summer", "2025-12-01", "2026-03-01"), ("Autumn", "2026-03-01", "2026-06-01"),
               ("Winter", "2026-06-01", "2026-09-01"), ("Spring", "2026-09-01", "2026-11-01")]
    for i, (name, a, b) in enumerate(seasons):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        ax.axvspan(a, b, ymin=0, ymax=1, color="#f3f2ee" if i % 2 == 0 else "#fafaf8", zorder=0)
        ax.text(a + (b - a) / 2, 3.15, name, ha="center", va="center", color=MUTED, fontsize=9)

    def bar(y, a, b, color, text, sub):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        ax.add_patch(Rectangle((mdates.date2num(a), y - 0.28), mdates.date2num(b) - mdates.date2num(a),
                               0.56, color=color, lw=0, zorder=2))
        ax.text(a, y + 0.47, text, ha="left", va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(a, y - 0.5, sub, ha="left", va="center", fontsize=8.5, color=INK2)

    bar(2, "2026-01-01", "2026-07-01", C["hybrid"], "Backtest (walk-forward)",
        "1 Jan - 30 Jun 2026, 181 days, issued daily from data to the previous midnight")
    issues = live.groupby("forecast_date")["issued_at"].first()
    first_live = pd.Timestamp(issues.index.min())
    last_live = pd.Timestamp(issues.index.max()) + pd.Timedelta(days=1)
    bar(1, first_live, last_live, C["naive"], "", "")
    ax.text(last_live, 1.47, "Live day-ahead track", ha="right", va="center", fontsize=10,
            fontweight="bold", color=INK)
    ax.text(pd.Timestamp("2026-07-05"), 1.0,
            f"from {first_live:%d %b}: issued the evening\nbefore, scored on AEMO actuals",
            ha="left", va="center", fontsize=8.5, color=INK2)
    # issue ticks
    for fday, iss in issues.items():
        x = pd.Timestamp(fday)
        ax.plot([x, x], [0.55, 0.72], color=INK, lw=1.0, zorder=3)
    ax.annotate(f"{len(issues)} issues to date\n(19:28 first, then 21:00 daily)",
                xy=(first_live, 0.6), xytext=(pd.Timestamp("2026-07-25"), 0.35),
                fontsize=8.5, color=INK2, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("", xy=(pd.Timestamp("2026-09-20"), 2.0), xytext=(pd.Timestamp("2026-07-05"), 2.0),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, ls=(0, (2, 3))))
    ax.text(pd.Timestamp("2026-08-12"), 2.18, "no seasonal overlap", ha="center",
            fontsize=8.5, color=INK2, style="italic")

    ax.set_xlim(start, end)
    ax.set_ylim(0.1, 3.4)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title("Two evaluation tracks: a summer-autumn backtest and a spring live track")
    save(fig, "fig02_evaluation_timeline.png")


# ------------------------------------------------------------------ fig 3
def fig03_rolling_rmse(bt: pd.DataFrame):
    sq = pd.DataFrame({m: (bt[f"{m}_pred"] - bt["actual"]) ** 2 for m in MODELS})
    daily_sum = sq.resample("D").sum(min_count=1)
    daily_n = sq.notna().resample("D").sum()
    roll = np.sqrt(daily_sum.rolling(30, min_periods=30).sum() / daily_n.rolling(30, min_periods=30).sum())
    roll = roll.dropna(how="all")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ends = {}
    for m in MODELS:
        ax.plot(roll.index, roll[m], color=C[m], lw=2 if m == "hybrid" else 1.6,
                label=LABEL[m], zorder=3 if m == "hybrid" else 2)
        ends[m] = np.log10(roll[m].dropna().iloc[-1])
    prev = -9
    for m, v in sorted(ends.items(), key=lambda kv: kv[1]):
        v = max(v, prev + 0.065)
        prev = v
        ax.text(roll.index[-1] + pd.Timedelta(days=1.5), 10 ** v, LABEL[m], color=INK2,
                fontsize=8.5, va="center")
    ax.set_yscale("log")
    ax.set_yticks([20, 50, 100, 200, 500, 1000])
    ax.set_yticklabels(["20", "50", "100", "200", "500", "1,000"])
    ax.set_ylabel("Trailing 30-day RMSE (A$/MWh, log scale)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(roll.index[0], roll.index[-1] + pd.Timedelta(days=14))
    above = (roll["hybrid"] > roll["naive"]).mean() * 100
    ax.set_title(f"The hybrid's rolling RMSE sits above its naive base in {above:.0f}% of windows")
    ax.text(0, 1.01, "Each point is the RMSE over the 30 days ending that day; "
            "January-February spikes dominate the early windows.",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    save(fig, "fig03_rolling_30day_rmse.png")


# ------------------------------------------------------------------ fig 4
def fig04_mae_by_hour(bt: pd.DataFrame):
    ae = pd.DataFrame({m: (bt[f"{m}_pred"] - bt["actual"]).abs() for m in MODELS})
    by_h = ae.groupby(ae.index.hour).mean()
    gap = by_h["hybrid"] - by_h["naive"]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.fill_between(by_h.index, by_h["naive"], by_h["hybrid"], color=C["hybrid"], alpha=0.10, lw=0)
    for m in MODELS:
        ax.plot(by_h.index, by_h[m], color=C[m], lw=2 if m == "hybrid" else 1.6, marker="o",
                ms=3.2, label=LABEL[m], zorder=3 if m == "hybrid" else 2)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0, None)
    ax.set_ylabel("Mean absolute error (A$/MWh)")
    ax.set_xlabel("Hour of day (NEM time, interval start)")
    ax.annotate(f"shaded: hybrid minus naive,\n{gap.min():.0f}-{gap.max():.0f} A$/MWh in every hour",
                xy=(3, (by_h.loc[3, "hybrid"] + by_h.loc[3, "naive"]) / 2),
                xytext=(1.0, 100), fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_title("The hybrid's error deficit is spread across the whole day")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5)
    save(fig, "fig04_mae_by_hour.png")
    return gap


# ------------------------------------------------------------------ fig 5
def fig05_daily_mean(bt: pd.DataFrame):
    cols = {"actual": "actual", **{m: f"{m}_pred" for m in MODELS}}
    daily = pd.DataFrame({k: bt[v] for k, v in cols.items()}).resample("D").mean()
    corr = (bt["hybrid_pred"] - bt["naive_pred"]).resample("D").mean()
    monthly = corr.resample("MS").mean()

    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                               gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    a.plot(daily.index, daily["actual"], color=INK, lw=2.2, label="Actual", zorder=4)
    for m in MODELS:
        a.plot(daily.index, daily[m], color=C[m], lw=1.9 if m == "hybrid" else 1.2,
               alpha=1 if m == "hybrid" else 0.85, label=LABEL[m], zorder=3 if m == "hybrid" else 2)
    a.set_yscale("symlog", linthresh=200, linscale=2.5)
    a.set_yticks([-100, 0, 100, 200, 500, 1000])
    a.set_yticklabels(["-100", "0", "100", "200", "500", "1,000"])
    a.set_ylim(-150, 1500)
    a.axhline(200, color=MUTED, lw=0.6, ls=(0, (2, 3)), zorder=0)
    a.set_ylabel("Daily mean price (A$/MWh,\nlinear to 200, log above)")
    a.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    a.set_title("Daily average price: actual vs each forecast, 1 Jan - 30 Jun 2026", pad=26)

    b.axhline(0, color=INK2, lw=0.8)
    pos = corr.clip(lower=0)
    neg = corr.clip(upper=0)
    b.bar(corr.index, pos, width=0.85, color=C["hybrid"], lw=0)
    b.bar(corr.index, neg, width=0.85, color=C["hybrid"], alpha=0.45, lw=0)
    for ms, v in monthly.items():
        me = ms + pd.offsets.MonthEnd(0)
        b.plot([ms, me], [v, v], color=INK, lw=1.6)
        b.text(ms + (me - ms) / 2, v + (8 if v >= 0 else -8), f"{v:+.0f}", ha="center",
               va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK, fontweight="bold")
    b.set_ylabel("Hybrid minus naive,\ndaily mean (A$/MWh)")
    b.text(0, 1.02, "The LSTM correction acts as a level offset (bars: daily; black lines: monthly mean)",
           transform=b.transAxes, fontsize=8.5, color=INK2, va="bottom")
    lim = np.nanpercentile(corr.abs(), 99) * 1.25
    b.set_ylim(-lim, lim)
    b.xaxis.set_major_locator(mdates.MonthLocator())
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    save(fig, "fig05_daily_mean_price.png")
    return monthly


# ------------------------------------------------------------------ overlays
def overlay(df, start, end, title, sub, name, log=False, linthresh=300, models=MODELS,
            notes=(), figsize=(10, 4.2), datefmt="%a %d %b"):
    w = df.loc[start:end]
    fig, ax = plt.subplots(figsize=figsize)
    for m in models:
        ax.plot(w.index, w[f"{m}_pred"], color=C[m], lw=1.9 if m == "hybrid" else 1.4,
                label=LABEL[m], zorder=3 if m == "hybrid" else 2,
                drawstyle="steps-post")
    actual = w["actual"] if "actual" in w else w["actual_price"]
    ax.plot(w.index, actual, color=INK, lw=2.2, label="Actual", zorder=5, drawstyle="steps-post")
    if log:
        symlog(ax, linthresh)
        ax.set_ylabel(PRICE + f", linear to {linthresh}, log above")
    else:
        ax.set_ylabel(PRICE)
    if datefmt:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(datefmt))
    ax.set_xlim(w.index[0], w.index[-1] + pd.Timedelta(minutes=30))
    for xy, xyt, txt in notes:
        ax.annotate(txt, xy=xy, xytext=xyt, fontsize=8.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.9))
    ax.set_title(title)
    ax.text(0, 1.01, sub, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    save(fig, name)
    return w


def week_rmse(w):
    # Per-model RMSE over window w (pandas mean, so NaN rows are skipped).
    return {m: np.sqrt(((w[f"{m}_pred"] - w["actual"]) ** 2).mean()) for m in MODELS}


def fig06_calm(bt):
    rmse = week_rmse(bt.loc["2026-04-06":"2026-04-12 23:30"])
    sub = "Week RMSE (A$/MWh): " + ", ".join(f"{LABEL[m]} {rmse[m]:.0f}" for m in MODELS)
    overlay(bt, "2026-04-06", "2026-04-12 23:30",
            "Calm week, 6-12 April: ARIMA stays flat while the others follow the daily shape",
            sub, "fig06_calm_week_april.png")


def fig07_spike(bt):
    rmse = week_rmse(bt.loc["2026-02-02":"2026-02-08 23:30"])
    sub = "Week RMSE (A$/MWh): " + ", ".join(f"{LABEL[m]} {rmse[m]:,.0f}" for m in MODELS)
    overlay(bt, "2026-02-02", "2026-02-08 23:30",
            "Spike week, 2-8 February: every model misses, each in its own way",
            sub, "fig07_spike_week_february.png", log=True)


def fig08_scatter(bt):
    lo, hi = -50, 400
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.12, top=0.8, bottom=0.18)
    for ax, m in zip(axes, MODELS):
        d = bt[["actual", f"{m}_pred"]].dropna()
        inside = d[(d["actual"].between(lo, hi)) & (d[f"{m}_pred"].between(lo, hi))]
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.scatter(inside["actual"], inside[f"{m}_pred"], s=3, color=C[m], alpha=0.18, lw=0,
                   rasterized=True, zorder=2)
        mae = (d[f"{m}_pred"] - d["actual"]).abs().mean()
        bias = (d[f"{m}_pred"] - d["actual"]).mean()
        ax.set_title(LABEL[m], fontsize=10, pad=6)
        ax.text(0.03, 0.97, f"MAE {mae:.1f}\nbias {bias:+.1f}\n{len(d) - len(inside)} pts off-axis",
                transform=ax.transAxes, va="top", fontsize=8, color=INK2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xticks([0, 100, 200, 300, 400])
        ax.set_yticks([0, 100, 200, 300, 400])
    axes[0].set_ylabel("Forecast (A$/MWh)")
    fig.text(0.5, 0.02, "Actual (A$/MWh)", ha="center", fontsize=9.5, color=INK2)
    fig.suptitle("Forecast vs actual, every half hour 1 Jan - 30 Jun 2026 (dashed line = perfect forecast)",
                 x=0.06, ha="left", fontsize=11, fontweight="bold", y=0.97)
    save(fig, "fig08_forecast_vs_actual_scatter.png")


def live_day(live, day):
    w = live[live["forecast_date"] == day].copy()
    w["actual"] = w["actual_price"]
    return w


def fig09_live_10oct(live):
    w = live_day(live, "2026-10-10")
    rm = week_rmse(w)
    t7 = pd.Timestamp("2026-10-10 07:00")
    t5 = pd.Timestamp("2026-10-10 05:00")
    issued = pd.Timestamp(w["issued_at"].iloc[0])
    overlay(w, w.index[0], w.index[-1],
            "Live, 10 October: one 07:00 spike no forecaster saw, close tracking otherwise",
            f"Issued {issued:%H:%M} on {issued:%d %b}. Day RMSE (A$/MWh): "
            + ", ".join(f"{LABEL[m]} {rm[m]:.0f}" for m in MODELS),
            "fig09_live_10oct.png", log=True, datefmt="%H:%M",
            notes=[((t7, w.loc[t7, "actual"]), (t7 + pd.Timedelta(hours=2), 9000),
                    f"07:00 settled at {w.loc[t7, 'actual']:,.0f}"),
                   ((t5, w.loc[t5, "hybrid_pred"]), (t5 - pd.Timedelta(hours=4.5), 2500),
                    "carried over from\nthe previous day's spike")])


def fig10_live_13oct(live):
    w = live_day(live, "2026-10-13")
    rm = week_rmse(w)
    issued = pd.Timestamp(w["issued_at"].iloc[0])
    t_h = w["hybrid_pred"].idxmax()
    t_p = w["aemo_pred"].idxmax()
    t_a = w["actual"].idxmax()
    overlay(w, w.index[0], w.index[-1],
            "Live, 13 October: the forecasters predicted two different spikes, and neither happened",
            f"Issued {issued:%H:%M} on {issued:%d %b}. Day RMSE (A$/MWh): "
            + ", ".join(f"{LABEL[m]} {rm[m]:,.0f}" for m in MODELS),
            "fig10_live_13oct.png", log=True, datefmt="%H:%M",
            notes=[((t_h, w.loc[t_h, "hybrid_pred"]), (t_h - pd.Timedelta(hours=8.5), 9000),
                    f"hybrid copies 12 Oct afternoon\n({w.loc[t_h, 'hybrid_pred']:,.0f} at {t_h:%H:%M})"),
                   ((t_p, w.loc[t_p, "aemo_pred"]), (t_p + pd.Timedelta(hours=1.2), 4500),
                    f"pre-dispatch scarcity\n({w.loc[t_p, 'aemo_pred']:,.0f} at {t_p:%H:%M})"),
                   ((t_a, w.loc[t_a, "actual"]), (t_a - pd.Timedelta(hours=7), 700),
                    f"actual peak only {w.loc[t_a, 'actual']:,.0f}")])


# ------------------------------------------------------------------ fig 11
def fig11_copy_forward(live):
    act = pd.read_csv(config.LIVE_PRICE_DEMAND_CSV, index_col=0,
                      parse_dates=True)["price"]
    a12 = act.loc["2026-10-12":"2026-10-12 23:30"]
    w = live_day(live, "2026-10-13")
    hrs12 = (a12.index - a12.index[0]).total_seconds() / 3600
    hrs13 = (w.index - w.index[0]).total_seconds() / 3600

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    panels = [
        ("1  Actual price, 12 Oct", [(hrs12, a12.values, INK, "Actual 12 Oct", 2.2)]),
        ("2  Naive base for 13 Oct (= 12 Oct)",
         [(hrs13, w["naive_pred"].values, C["naive"], "Naive base", 2.0)]),
        ("3  13 Oct: hybrid forecast vs actual",
         [(hrs13, w["hybrid_pred"].values, C["hybrid"], "Hybrid = naive + LSTM", 2.0),
          (hrs13, w["actual"].values, INK, "Actual 13 Oct", 2.2)]),
    ]
    for ax, (title, series) in zip(axes, panels):
        for x, y, col, lab, lw in series:
            ax.plot(x, y, color=col, lw=lw, label=lab, drawstyle="steps-post")
        symlog(ax)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(["00", "06", "12", "18", "24"])
        ax.set_xlim(0, 24)
        ax.axvspan(14, 16, color=C["hybrid"], alpha=0.07, lw=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Hour")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel(PRICE)
    peak = a12.idxmax()
    # Label sits right of the peak, clear of the legend in the top left.
    axes[0].annotate(f"{a12.max():,.0f}\nat {peak:%H:%M}",
                     xy=(15, a12.max()), xytext=(17.3, 2500), fontsize=8.5, color=INK2,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    lstm = (w["hybrid_pred"] - w["naive_pred"]).loc["2026-10-13 14:00":"2026-10-13 15:30"]
    axes[2].text(0.98, 0.04, f"LSTM adds only {lstm.min():+.0f} "
                 f"to {lstm.max():+.0f}\nin the 14:00-15:30 window",
                 transform=axes[2].transAxes, ha="right", fontsize=8, color=INK2)
    fig.subplots_adjust(wspace=0.32, top=0.8)
    for i in range(2):
        x0 = axes[i].get_position().x1 + 0.008
        x1 = axes[i + 1].get_position().x0 - 0.012
        fig.add_artist(FancyArrowPatch((x0, 0.5), (x1, 0.5),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=16, color=INK2, lw=1.4))
    fig.suptitle("How a spike is copied forward: yesterday's actual becomes today's forecast",
                 x=0.06, ha="left", fontsize=11, fontweight="bold", y=0.98)
    save(fig, "fig11_copy_forward_mechanism.png")


def main():
    bt = load_backtest()
    live = load_live()
    print(f"backtest {bt.index.min()} .. {bt.index.max()} ({len(bt)} rows, "
          f"{bt['aemo_pred'].notna().sum()} with pre-dispatch)")
    fig02_timeline(live)
    fig03_rolling_rmse(bt)
    gap = fig04_mae_by_hour(bt)
    print(f"  hybrid-naive MAE gap by hour: {gap.min():.1f} .. {gap.max():.1f}")
    monthly = fig05_daily_mean(bt)
    print("  monthly LSTM offset:", monthly.round(1).to_dict())
    fig06_calm(bt)
    fig07_spike(bt)
    fig08_scatter(bt)
    fig09_live_10oct(live)
    fig10_live_13oct(live)
    fig11_copy_forward(live)


if __name__ == "__main__":
    main()
