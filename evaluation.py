"""
Evaluation: point metrics, normal-vs-spike breakdown, ARIMA-vs-Hybrid table,
and a fixed set of plots saved to /outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


# ----------------------------------------------------------------------------- #
# Metrics
# ----------------------------------------------------------------------------- #
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """
    MAPE with a floor on the denominator to handle zero / near-zero / negative
    NEM prices. We use max(|y_true|, eps) so a 1 AUD/MWh floor avoids blowing
    up around zero crossings. Returns a percentage.
    """
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


@dataclass
class MetricSet:
    rmse: float
    mae: float
    mape: float

    def as_row(self) -> Dict[str, float]:
        return {"RMSE": self.rmse, "MAE": self.mae, "MAPE_%": self.mape}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricSet:
    return MetricSet(rmse=rmse(y_true, y_pred), mae=mae(y_true, y_pred), mape=mape(y_true, y_pred))


# ----------------------------------------------------------------------------- #
# Reporting
# ----------------------------------------------------------------------------- #
def _print_table(title: str, df: pd.DataFrame) -> None:
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)
    print(df.round(3).to_string())
    print("-" * 70)


def build_results_table(
    y_test: np.ndarray,
    arima_pred: np.ndarray,
    hybrid_pred: np.ndarray,
    spike_mask: np.ndarray,
) -> pd.DataFrame:
    """
    Build a tidy comparison table:
        rows    : ARIMA, Hybrid
        columns : RMSE, MAE, MAPE  (overall, normal, spike)
    """
    df = pd.DataFrame({
        "actual": y_test,
        "ARIMA_pred": arima_pred,
        "ARIMA-LSTM Hybrid_pred": hybrid_pred,
        "is_spike": spike_mask.astype(bool).astype(int),
    })
    return _segmented_metrics(df, ["ARIMA", "ARIMA-LSTM Hybrid"])


def export_predictions(
    idx_test: pd.DatetimeIndex,
    y_true: np.ndarray,
    arima_pred: np.ndarray,
    hybrid_pred: np.ndarray,
    spike_mask: np.ndarray,
    out_path: Path,
) -> pd.DataFrame:
    """
    Write the raw timestamped actual-vs-predicted series backing the overlay
    and scatter figures, so the plots can be reproduced or re-styled without
    re-running the pipeline. One row per test timestamp.
    """
    df = pd.DataFrame(
        {
            "actual": np.asarray(y_true),
            "arima_pred": np.asarray(arima_pred),
            "hybrid_pred": np.asarray(hybrid_pred),
            "arima_residual": np.asarray(y_true) - np.asarray(arima_pred),
            "hybrid_residual": np.asarray(y_true) - np.asarray(hybrid_pred),
            "is_spike": np.asarray(spike_mask).astype(int),
        },
        index=pd.DatetimeIndex(idx_test, name="timestamp"),
    )
    df.to_csv(out_path)
    return df


def print_timing_table(timings: Dict[str, float]) -> None:
    df = pd.DataFrame(
        [{"component": k, "seconds": v} for k, v in timings.items()]
    ).set_index("component")
    _print_table("Computational cost (training time)", df)


# ----------------------------------------------------------------------------- #
# Plots
# ----------------------------------------------------------------------------- #
def _ensure_outdir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_full_test(
    idx: pd.DatetimeIndex,
    y_true: np.ndarray,
    arima_pred: np.ndarray,
    hybrid_pred: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE, dpi=config.PLOT_DPI)
    ax.plot(idx, y_true, label="Actual", color="black", linewidth=0.9)
    ax.plot(idx, arima_pred, label="ARIMA", color="tab:orange", linewidth=0.7, alpha=0.8)
    ax.plot(idx, hybrid_pred, label="Hybrid", color="tab:blue", linewidth=0.7, alpha=0.9)
    ax.set_title("Test set: actual vs predicted price")
    ax.set_ylabel("Price (AUD/MWh)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_zoom_week(
    idx: pd.DatetimeIndex,
    y_true: np.ndarray,
    arima_pred: np.ndarray,
    hybrid_pred: np.ndarray,
    out_path: Path,
) -> None:
    """One-week window centred on the highest-volatility region of the test set."""
    if len(idx) == 0:
        return
    window = 7 * config.PERIODS_PER_DAY
    if len(idx) <= window:
        start, end = 0, len(idx)
    else:
        # Pick the week with the highest variance for an informative zoom
        rolling_var = pd.Series(y_true).rolling(window).var()
        center = int(rolling_var.idxmax() if not rolling_var.isna().all() else len(idx) // 2)
        start = max(0, center - window // 2)
        end = min(len(idx), start + window)

    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE, dpi=config.PLOT_DPI)
    ax.plot(idx[start:end], y_true[start:end], label="Actual", color="black")
    ax.plot(idx[start:end], arima_pred[start:end], label="ARIMA", color="tab:orange")
    ax.plot(idx[start:end], hybrid_pred[start:end], label="Hybrid", color="tab:blue")
    ax.set_title("Zoom: 1-week window (highest-variance region of test set)")
    ax.set_ylabel("Price (AUD/MWh)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_residual_hist(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str
) -> None:
    import seaborn as sns

    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 5), dpi=config.PLOT_DPI)
    sns.histplot(residuals, bins=80, kde=True, ax=ax, color="tab:blue")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Residual (AUD/MWh)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scatter(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=config.PLOT_DPI)
    ax.scatter(y_true, y_pred, s=4, alpha=0.3)
    lim_lo = min(y_true.min(), y_pred.min())
    lim_hi = max(y_true.max(), y_pred.max())
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual (AUD/MWh)")
    ax.set_ylabel("Predicted (AUD/MWh)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_spike_overlay(
    idx: pd.DatetimeIndex,
    y_true: np.ndarray,
    hybrid_pred: np.ndarray,
    spike_mask: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE, dpi=config.PLOT_DPI)
    ax.plot(idx, y_true, color="black", linewidth=0.9, label="Actual")
    ax.plot(idx, hybrid_pred, color="tab:blue", linewidth=0.7, alpha=0.9, label="Hybrid")
    if spike_mask.any():
        spike_idx = idx[spike_mask.astype(bool)]
        ax.scatter(
            spike_idx,
            y_true[spike_mask.astype(bool)],
            color="red",
            s=10,
            label="Spike period",
            zorder=5,
        )
    ax.set_title("Spike-period performance (test set)")
    ax.set_ylabel("Price (AUD/MWh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_metric_bars(results_df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["overall_RMSE", "overall_MAE", "overall_MAPE_%"]
    short = ["RMSE", "MAE", "MAPE %"]
    sub = results_df[metrics].copy()
    sub.columns = short

    fig, ax = plt.subplots(figsize=(8, 5), dpi=config.PLOT_DPI)
    sub.T.plot(kind="bar", ax=ax)
    ax.set_title("ARIMA vs Hybrid — overall metrics")
    ax.set_ylabel("Error")
    ax.set_xticklabels(short, rotation=0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------------- #
# Public entry point
# ----------------------------------------------------------------------------- #
def evaluate(
    idx_test: pd.DatetimeIndex,
    y_test: np.ndarray,
    arima_pred: np.ndarray,
    hybrid_pred: np.ndarray,
    spike_test: np.ndarray,
    timings: Dict[str, float],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Compute metrics, print tables, save plots. Returns the comparison DataFrame.
    """
    out_dir = _ensure_outdir(out_dir)

    results_df = build_results_table(y_test, arima_pred, hybrid_pred, spike_test)
    _print_table("Forecast accuracy (test set)", results_df)
    print_timing_table(timings)

    plot_full_test(idx_test, y_test, arima_pred, hybrid_pred, out_dir / "test_full.png")
    plot_zoom_week(idx_test, y_test, arima_pred, hybrid_pred, out_dir / "test_zoom_week.png")
    plot_residual_hist(y_test, hybrid_pred, out_dir / "residuals_hybrid.png",
                       "Hybrid residuals (test set)")
    plot_residual_hist(y_test, arima_pred, out_dir / "residuals_arima.png",
                       "ARIMA residuals (test set)")
    plot_scatter(y_test, hybrid_pred, out_dir / "scatter_hybrid.png",
                 "Hybrid: actual vs predicted")
    plot_scatter(y_test, arima_pred, out_dir / "scatter_arima.png",
                 "ARIMA: actual vs predicted")
    plot_spike_overlay(idx_test, y_test, hybrid_pred, spike_test,
                       out_dir / "spike_overlay.png")
    plot_metric_bars(results_df, out_dir / "metric_bars.png")

    results_df.to_csv(out_dir / "results.csv")
    export_predictions(
        idx_test, y_test, arima_pred, hybrid_pred, spike_test,
        out_dir / "predictions.csv",
    )
    print(f"\n[evaluation] Plots and results saved to {out_dir}")
    return results_df


# ----------------------------------------------------------------------------- #
# Walk-forward aggregation (headline evaluation path)
# ----------------------------------------------------------------------------- #
def _segmented_metrics(df: pd.DataFrame, models: list) -> pd.DataFrame:
    """Rows = model, columns = overall / normal / spike RMSE, MAE, MAPE."""
    rows = {}
    for name in models:
        pred = df[f"{name}_pred"]
        row = {}
        for seg, mask in [
            ("overall", np.ones(len(df), dtype=bool)),
            ("normal", df["is_spike"].values == 0),
            ("spike", df["is_spike"].values == 1),
        ]:
            if mask.sum() == 0:
                row.update({f"{seg}_RMSE": np.nan, f"{seg}_MAE": np.nan, f"{seg}_MAPE_%": np.nan})
                continue
            m = compute_metrics(df["actual"].values[mask], pred.values[mask]).as_row()
            row.update({f"{seg}_{k}": v for k, v in m.items()})
        rows[name] = row
    return pd.DataFrame(rows).T


def _grouped(df: pd.DataFrame, models: list, key, fn) -> pd.DataFrame:
    """Per-group `fn(actual, pred)` for each model: rows = groups, columns = models."""
    return pd.DataFrame({
        name: df.groupby(key).apply(
            lambda d, name=name: fn(d["actual"].values, d[f"{name}_pred"].values)
        )
        for name in models
    })


def paired_t_test(daily: pd.DataFrame, a: str, b: str) -> dict:
    """One-sample t-test on the daily RMSE differences (model a - model b)."""
    from scipy import stats

    diff = daily[a].values - daily[b].values
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return {"mean_diff": np.nan, "t": np.nan, "p": np.nan, "n": len(diff)}
    t, p = stats.ttest_1samp(diff, 0.0)
    return {"mean_diff": float(diff.mean()), "t": float(t), "p": float(p), "n": len(diff)}


def evaluate_walkforward(
    predictions: pd.DataFrame,
    target: str,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Aggregate the tidy walk-forward predictions into the report tables/figures:
      - headline model comparison (Naive / ARIMA / Hybrid) with spike segments
      - monthly + weekday/weekend breakdown
      - error-by-hour-of-day profile
      - rolling 30-day RMSE per model
      - paired t-test of daily RMSE differences (hybrid vs ARIMA, hybrid vs naive)
    """
    out_dir = _ensure_outdir(out_dir)
    df = predictions.copy()
    df["day"] = df.index.normalize()

    models = [m for m in ("naive", "arima", "hybrid") if f"{m}_pred" in df.columns]

    # 1) Headline table with spike/peak segments
    table = _segmented_metrics(df, models)
    _print_table(f"Day-ahead forecast accuracy — target: {target}", table)
    table.to_csv(out_dir / f"results_{target}.csv")

    # 2) Monthly breakdown
    monthly_df = _grouped(df, models, df.index.to_period("M"), rmse)
    _print_table("RMSE by month", monthly_df)
    monthly_df.to_csv(out_dir / f"monthly_rmse_{target}.csv")

    # 3) Weekday / weekend breakdown
    dow = df["day"].dt.dayofweek
    seg = pd.Series(np.where(dow < 5, "weekday", "weekend"), index=df.index, name="seg")
    wd = {}
    for name in models:
        wd[name] = {
            s: rmse(df["actual"].values[mask], df[f"{name}_pred"].values[mask])
            for s, mask in [("weekday", (seg == "weekday").values), ("weekend", (seg == "weekend").values)]
        }
    _print_table("RMSE by weekday/weekend", pd.DataFrame(wd))

    # 4) Error-by-hour-of-day profile (hybrid + arima)
    df["hour"] = df.index.hour
    hourly = _grouped(df, models, "hour", mae)
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE, dpi=config.PLOT_DPI)
    hourly.plot(ax=ax, marker="o", markersize=3)
    ax.set_title(f"Mean absolute error by hour of day — {target}")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig.savefig(out_dir / f"error_by_hour_{target}.png")
    plt.close(fig)

    # 5) Rolling 30-day RMSE per model
    daily = _grouped(df, models, "day", rmse)
    roll = daily.rolling(30, min_periods=7).mean()
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE, dpi=config.PLOT_DPI)
    roll.plot(ax=ax, linewidth=1.2)
    ax.set_title(f"Rolling 30-day RMSE — {target}")
    ax.set_ylabel("RMSE")
    fig.tight_layout()
    fig.savefig(out_dir / f"rolling_rmse_{target}.png")
    plt.close(fig)

    # 6) Paired significance tests on daily RMSE differences
    print("\nPaired t-tests on daily RMSE differences (a - b):")
    for a, b in [("hybrid", "arima"), ("hybrid", "naive"), ("arima", "naive")]:
        if a in daily.columns and b in daily.columns:
            r = paired_t_test(daily, a, b)
            print(
                f"  {a:>7} vs {b:<7}: mean_diff={r['mean_diff']:+.3f}  "
                f"t={r['t']:.2f}  p={r['p']:.4f}  n={r['n']}"
            )

    # 7) Forecast-vs-actual overlay for the whole eval window
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE, dpi=config.PLOT_DPI)
    ax.plot(df.index, df["actual"], label="Actual", color="black", linewidth=0.6)
    ax.plot(df.index, df["arima_pred"], label="ARIMA", color="tab:orange", linewidth=0.5, alpha=0.8)
    ax.plot(df.index, df["hybrid_pred"], label="Hybrid", color="tab:blue", linewidth=0.5, alpha=0.9)
    ax.set_title(f"Day-ahead: actual vs predicted — {target}")
    ax.set_ylabel(target)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"test_full_{target}.png")
    plt.close(fig)

    print(f"\n[evaluation] Walk-forward results saved to {out_dir}")
    return table
