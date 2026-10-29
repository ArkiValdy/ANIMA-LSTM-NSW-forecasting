"""
Operational day-ahead forecast: the walk-forward protocol applied to the
real next day instead of a historical one.

At the end of the last fully observed day D, train on everything up to D and
forecast all 48 periods of D+1 with the seasonal naive, ARIMA and hybrid
models. Unlike the evaluation runs, forecast-day temperature is a genuine
Open-Meteo weather FORECAST, not the observed temperature.

Each run also scores every earlier forecast whose day now has complete
actuals, so running it daily builds up a true out-of-sample track record.

    python main.py --mode forecast --target demand --refresh

Outputs (outputs/forecast/):
  forecast_{target}_{date}.csv   48-period forecast issued before the day
  forecast_{target}_{date}.png   forecast vs the two preceding days
  scores_{target}.csv            per-day RMSE/MAE of scored forecasts
  arima_order_{target}.json      cached auto_arima order (quarterly refresh)
  hindcast_{target}_{date}.csv   --forecast_date in the past; never scored
                                 (its weather is not an ex-ante forecast)
"""

from __future__ import annotations

import functools
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import arima_model
import data_loader
import evaluation
import lstm_model
import preprocessing
import walk_forward

if str(config.DATA_DIR) not in sys.path:
    sys.path.append(str(config.DATA_DIR))
import download_temperature  # noqa: E402
from download_temperature import LAT, LON  # noqa: E402,F401  (the same Sydney point)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PERIODS = config.PERIODS_PER_DAY
MODELS = ("naive", "arima", "hybrid")
UNIT = {"demand": "MW", "price": "$/MWh"}


# ----------------------------------------------------------------------------- #
# Weather forecast
# ----------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=8)
def _fetch_weather_forecast(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return download_temperature.fetch(
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), url=FORECAST_URL, timeout=60
    ).dropna()


def fetch_weather_forecast(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Hourly Open-Meteo forecast temperature (Sydney local time), inclusive dates.
    Recent past dates come from the forecast model's own analysis, which fills
    the few days the historical archive has not published yet. Cached per
    process, so dashboard.py's price and demand runs share one request."""
    return _fetch_weather_forecast(start, end).copy()


def _merge_temperature(archive_path: Path, fc: pd.DataFrame, out_path: Path) -> Path:
    """Archive observations where available, forecast-model values elsewhere."""
    archive = pd.read_csv(archive_path, index_col=0, parse_dates=True).dropna()
    merged = pd.concat([archive, fc[~fc.index.isin(archive.index)]]).sort_index()
    merged.index.name = "timestamp"
    merged.to_csv(out_path)
    return out_path


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def last_complete_day(price_demand_path: Path, target: str) -> pd.Timestamp:
    """Latest calendar day with all 48 periods of the target observed."""
    pdf = data_loader._load_consolidated(price_demand_path)
    counts = pdf[target].notna().groupby(pdf.index.normalize()).sum()
    full = counts[counts == PERIODS]
    if full.empty:
        raise ValueError(f"No complete day of {target} in {price_demand_path}")
    return full.index.max()


def _arima_order(train_series: pd.Series, fday: pd.Timestamp, target: str, out_dir: Path) -> tuple:
    """Cached auto_arima order, re-selected every ARIMA_ORDER_REFIT_DAYS."""
    path = out_dir / f"arima_order_{target}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        age = (fday - pd.Timestamp(cached["selected_for"])).days
        if 0 <= age < config.ARIMA_ORDER_REFIT_DAYS:
            print(f"[forecast] ARIMA order {tuple(cached['order'])} (cached, {age} days old)")
            return tuple(cached["order"])
    print("[forecast] selecting ARIMA order (auto_arima, may take a few minutes)...")
    order = arima_model.select_order_scaled(train_series.values)
    path.write_text(json.dumps({"order": list(order), "selected_for": str(fday.date())}))
    return order


# ----------------------------------------------------------------------------- #
# Scoring of earlier forecasts
# ----------------------------------------------------------------------------- #
def score_forecasts(actuals: pd.Series, target: str, out_dir: Path) -> pd.DataFrame | None:
    """Score every forecast_{target}_*.csv whose day has 48 observed actuals."""
    rows = []
    for f in sorted(out_dir.glob(f"forecast_{target}_*.csv")):
        fc = pd.read_csv(f, index_col=0, parse_dates=True)
        act = actuals.reindex(fc.index)
        if act.notna().sum() != PERIODS:
            continue
        row = {"date": fc.index[0].date(), "issued_at": fc["issued_at"].iloc[0]}
        for m in MODELS:
            pred = fc[f"{m}_pred"].values
            row[f"{m}_RMSE"] = evaluation.rmse(act.values, pred)
            row[f"{m}_MAE"] = evaluation.mae(act.values, pred)
        rows.append(row)
    if not rows:
        print("[forecast] no earlier forecasts with complete actuals to score yet")
        return None

    scores = pd.DataFrame(rows).set_index("date")
    scores.to_csv(out_dir / f"scores_{target}.csv")
    print(f"\n[forecast] scored {len(scores)} past forecast day(s) — {target} RMSE:")
    print(scores[[f"{m}_RMSE" for m in MODELS]].round(2).to_string())
    if len(scores) > 1:
        print("  mean   " + "  ".join(
            f"{m} {scores[f'{m}_RMSE'].mean():.2f}" for m in MODELS))
    return scores


# ----------------------------------------------------------------------------- #
# Forecast
# ----------------------------------------------------------------------------- #
def _plot(fc: pd.DataFrame, recent: pd.Series, target: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(recent.index, recent.values, color="black", lw=1.2, label="actual (last 2 days)")
    for m, c in zip(MODELS, ("tab:gray", "tab:blue", "tab:red")):
        ax.plot(fc.index, fc[f"{m}_pred"], color=c, lw=1.4, label=m)
    ax.axvline(fc.index[0], color="k", ls=":", lw=0.8)
    ax.set_title(f"NSW {target} day-ahead forecast for {fc.index[0].date()}")
    ax.set_ylabel(f"{target} ({UNIT[target]})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_forecast(
    price_demand_path: Path,
    temperature_path: Path,
    target: str,
    forecast_date: str | None = None,
    out_dir: Path = config.OUTPUT_DIR / "forecast",
    score: bool = True,
    extra_columns: dict | None = None,
) -> pd.DataFrame:
    """Forecast fday (default: the day after the last complete day) and save it.
    extra_columns are appended to the saved frame after the model columns (a
    Series is aligned on the forecast index), e.g. dashboard.py's aemo_pd."""
    config.TARGET_COLUMN = target
    out_dir.mkdir(parents=True, exist_ok=True)
    issued_at = datetime.now().isoformat(timespec="seconds")

    # ---- Which day? ---------------------------------------------------------- #
    last_day = last_complete_day(Path(price_demand_path), target)
    next_day = last_day + pd.Timedelta(days=1)
    fday = pd.Timestamp(forecast_date) if forecast_date else next_day
    if fday > next_day:
        raise SystemExit(
            f"Cannot forecast {fday.date()}: data is only complete through "
            f"{last_day.date()}, so the next day-ahead forecast is {next_day.date()}."
        )
    hindcast = fday < next_day
    print(f"\n[forecast] data complete through {last_day.date()} -> forecasting {fday.date()}"
          f"{'  (HINDCAST: past day, not scored)' if hindcast else ''}")

    # ---- Weather: archive + forecast model ------------------------------------ #
    wx = fetch_weather_forecast(fday - pd.Timedelta(days=10), fday + pd.Timedelta(days=1))
    temp_path = _merge_temperature(Path(temperature_path), wx, out_dir / "temperature_with_forecast.csv")

    # ---- History up to the end of the day before fday ------------------------- #
    df = data_loader.load_dataset(price_demand_path, temp_path, verbose=False)
    actuals_all = df[target]
    df = df[df.index < fday]
    expected_end = fday - pd.Timedelta(minutes=30)
    if df.index.max() != expected_end:
        raise SystemExit(f"History ends at {df.index.max()}, expected {expected_end}")

    # score=False when price_demand_path is not pure actuals (dashboard.py fills
    # the rest of today with AEMO pre-dispatch); the caller scores separately.
    if score and not hindcast:
        score_forecasts(actuals_all, target, out_dir)

    # Forecast-day rows: temperature from the forecast, target unknown
    temp30 = data_loader.load_temperature(temp_path).interpolate(limit_area="inside")
    fidx = pd.date_range(fday, periods=PERIODS, freq="30min")
    fday_temp = temp30["temperature"].reindex(fidx)
    if fday_temp.isna().any():
        raise SystemExit(f"Weather forecast missing for {fday.date()}")
    future = pd.DataFrame({"price": np.nan, "demand": np.nan, "temperature": fday_temp.values}, index=fidx)

    feats = preprocessing.build_features(pd.concat([df, future]), forecast_day=fday)
    fday_feat = walk_forward._day_slice(feats, fday)
    prev_feat = walk_forward._day_slice(feats, fday - pd.Timedelta(days=1))
    if len(fday_feat) != PERIODS or len(prev_feat) != PERIODS:
        raise SystemExit("Forecast day or previous day is incomplete after feature building")

    # ---- Models (same protocol as one walk-forward step) ---------------------- #
    target_raw = feats.loc[feats.index < fday, target].astype(float)
    train_start = expected_end - pd.Timedelta(days=config.TRAIN_DAYS) + pd.Timedelta(minutes=30)
    train_series = target_raw.loc[target_raw.index >= train_start]

    order = _arima_order(train_series, fday, target, out_dir)
    arima = walk_forward.ArimaDayAhead()
    arima.refit(train_series.values, order)
    arima_pred = arima.predict()

    naive_pred = target_raw[prev_feat.index].values

    lstm_state = walk_forward.fit_residual_lstm(
        feats, target_raw, order, expected_end,
        weights_path=out_dir / f"lstm_{target}.weights.h5",
    )
    X_seq, X_ctx = preprocessing.build_forecast_sequence(prev_feat, fday_feat)
    lstm_resid = lstm_model.predict_sequence(lstm_state, X_seq, X_ctx)
    hybrid_pred = walk_forward.hybrid_base(naive_pred, arima_pred) + lstm_resid

    # ---- Save ------------------------------------------------------------------ #
    fc = pd.DataFrame(
        {
            "temperature_fc": fday_temp.values,
            "naive_pred": naive_pred,
            "arima_pred": arima_pred,
            "lstm_resid": lstm_resid,
            "hybrid_pred": hybrid_pred,
            "issued_at": issued_at,
            "data_through": str(last_day.date()),
        },
        index=pd.DatetimeIndex(fidx, name="timestamp"),
    )
    for name, values in (extra_columns or {}).items():
        fc[name] = values
    stem = f"{'hindcast' if hindcast else 'forecast'}_{target}_{fday.date()}"
    fc.to_csv(out_dir / f"{stem}.csv")
    recent = target_raw[target_raw.index >= fday - pd.Timedelta(days=2)]
    _plot(fc, recent, target, out_dir / f"{stem}.png")

    print(f"\n[forecast] {target} for {fday.date()} ({UNIT[target]}), ARIMA order {order}:")
    summary = fc[[f"{m}_pred" for m in MODELS]].agg(["mean", "min", "max"]).T
    summary.index = list(MODELS)
    print(summary.round(1).to_string())
    print(f"[forecast] saved {out_dir / (stem + '.csv')}")
    return fc
