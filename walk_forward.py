"""
Rolling-origin day-ahead walk-forward engine (plan Sec 3).

Protocol: at the end of day D, using the previous TRAIN_DAYS days of
half-hourly data, forecast all 48 periods of day D+1. Slide forward one day
and repeat over the evaluation window.

Refit schedule (compute reality):
  - ARIMA order selection: quarterly (auto_arima)
  - ARIMA parameter refit: weekly (fixed-order fit on the current window)
  - ARIMA daily: state update with the observed day (maxiter=0)
  - LSTM: monthly warm-started refit on day-ahead ARIMA residuals

Baselines: seasonal naive (same period yesterday) + ARIMA only + hybrid.

Outputs (per target):
  outputs/walkforward/predictions_{target}.csv  — tidy per-period frame
  outputs/walkforward/timings_{target}.csv      — component wall-clock
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
import arima_model
import lstm_model
import preprocessing


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame        # per-period tidy frame
    timings: pd.DataFrame            # component wall-clock
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    n_days: int


# ----------------------------------------------------------------------------- #
# ARIMA day-ahead state
# ----------------------------------------------------------------------------- #
class ArimaDayAhead:
    """Holds the persistent ARIMA state across forecast days."""

    def __init__(self) -> None:
        self.order: tuple | None = None
        self.model = None
        self.scaler: StandardScaler | None = None
        self.last_refit_day: pd.Timestamp | None = None

    def refit(self, target_raw: np.ndarray, order: tuple) -> float:
        """Weekly refit: fit scaler + fixed-order ARIMA on the training window."""
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(target_raw.reshape(-1, 1)).ravel()
        t0 = time.perf_counter()
        self.model = arima_model.fit_fixed_order(scaled, order)
        dt = time.perf_counter() - t0
        self.order = order
        return dt

    def update(self, day_actual_raw: np.ndarray) -> None:
        """Absorb one observed day into the model state (no param refit)."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("ArimaDayAhead.update called before refit")
        day_scaled = self.scaler.transform(np.asarray(day_actual_raw).reshape(-1, 1)).ravel()
        arima_model.update_with_day(self.model, day_scaled)

    def predict(self) -> np.ndarray:
        """Forecast the full next day; returns RAW units."""
        pred_scaled = arima_model.predict_day_ahead(self.model, config.HORIZON_PERIODS)
        return self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()


# ----------------------------------------------------------------------------- #
# LSTM residual training data (day-ahead residuals over history)
# ----------------------------------------------------------------------------- #
def build_residual_history(
    target_raw: pd.Series,
    order: tuple,
    start: pd.Timestamp,
    end: pd.Timestamp,
    base: str = config.HYBRID_BASE,
) -> pd.Series:
    """
    Replay the day-ahead base protocol over [start, end] (inclusive end),
    returning the per-period raw residuals aligned to each period's timestamp.

    base="arima": ARIMA re-fitted once per ARIMA_REFIT_DAYS on all history up
        to that point, then each day within the refit window is forecast 48
        periods ahead. Residual at period p of day D+1 = actual - ARIMA
        forecast made at end of day D (honest day-ahead).
    base="naive": residual = actual - same-period-yesterday (no model replay).
    """
    periods = config.PERIODS_PER_DAY
    refit_days = config.ARIMA_REFIT_DAYS

    day_index = target_raw.index.normalize()
    days = pd.unique(day_index)
    days = pd.DatetimeIndex(sorted(days))
    days = days[(days >= start.normalize()) & (days <= end.normalize())]

    if base == "naive":
        # same-period-yesterday residual: shift by exactly one day
        naive = target_raw.shift(periods)
        mask = target_raw.index.normalize().isin(days)
        resid = (target_raw - naive).loc[mask]
        resid = resid.dropna()
        if resid.empty:
            raise ValueError("build_residual_history (naive) produced no days")
        return resid.sort_index()

    residual_series = []
    # Partition the replay days into refit blocks of ARIMA_REFIT_DAYS
    for block_start_i in range(0, len(days), refit_days):
        block_days = days[block_start_i : block_start_i + refit_days]
        # Fit on ALL history up to the day before the block begins
        fit_end = days[block_start_i] - pd.Timedelta(minutes=30)
        fit_window = target_raw.loc[:fit_end]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(fit_window.values.reshape(-1, 1)).ravel()
        model = arima_model.fit_fixed_order(scaled, order)

        for day in block_days:
            day_mask = day_index == day
            actual_raw = target_raw[day_mask].values
            if len(actual_raw) != periods:
                continue
            pred_scaled = arima_model.predict_day_ahead(model, periods)
            pred_raw = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            residual_series.append(
                pd.Series(actual_raw - pred_raw, index=target_raw[day_mask].index)
            )

    if not residual_series:
        raise ValueError("build_residual_history produced no days")
    return pd.concat(residual_series).sort_index()


def fit_residual_lstm(
    df_features: pd.DataFrame,
    target_raw: pd.Series,
    order: tuple,
    train_end: pd.Timestamp,
    weights_path: Path | None = None,
    initial_weights: list[np.ndarray] | None = None,
) -> lstm_model.LstmSequenceModel:
    """Train the LSTM on the last LSTM_RESIDUAL_DAYS of base-forecast residuals
    ending at `train_end` (inclusive). Shared by walk-forward and forecast mode."""
    resid_start = train_end - pd.Timedelta(days=config.LSTM_RESIDUAL_DAYS) + pd.Timedelta(minutes=30)
    residuals = build_residual_history(
        target_raw, order, resid_start, train_end, base=config.HYBRID_BASE
    )
    # SequenceSet over the same window (features + residuals)
    win = df_features.loc[
        (df_features.index >= resid_start) & (df_features.index <= train_end)
    ].copy()
    win["_residual"] = residuals.reindex(win.index)
    win = win.dropna(subset=["_residual"])

    X_seq, X_ctx, y_resid, _, _ = preprocessing.build_sequences(win, target="_residual")

    # hold out the most recent ~30 days for early stopping
    n_val = min(30, int(len(X_seq) * 0.15))
    if n_val > 0:
        Xs_tr, Xc_tr, yr_tr = X_seq[:-n_val], X_ctx[:-n_val], y_resid[:-n_val]
        Xs_va, Xc_va, yr_va = X_seq[-n_val:], X_ctx[-n_val:], y_resid[-n_val:]
    else:
        Xs_tr, Xc_tr, yr_tr = X_seq, X_ctx, y_resid
        Xs_va, Xc_va, yr_va = X_seq[-1:], X_ctx[-1:], y_resid[-1:]

    return lstm_model.train_sequence_model(
        Xs_tr, Xc_tr, yr_tr,
        Xs_va, Xc_va, yr_va,
        weights_path=weights_path,
        initial_weights=initial_weights,
        verbose=0,
    )


# ----------------------------------------------------------------------------- #
# Main walk-forward loop
# ----------------------------------------------------------------------------- #
def _day_slice(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    mask = df.index.normalize() == day.normalize()
    return df.loc[mask]


def hybrid_base(naive_pred: np.ndarray, arima_pred: np.ndarray) -> np.ndarray:
    """Base forecast the LSTM residual correction is added to (config.HYBRID_BASE)."""
    # "naive": hybrid = seasonal naive + LSTM correction (weather/weekday signal)
    return naive_pred if config.HYBRID_BASE == "naive" else arima_pred


def run_walkforward(
    df_features: pd.DataFrame,
    target: str | None = None,
    eval_start: str = config.EVAL_START,
    max_eval_days: int | None = None,
    out_dir: Path = config.OUTPUT_DIR / "walkforward",
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Run the rolling day-ahead evaluation for one target.

    df_features: feature frame from preprocessing.build_features (index =
                 half-hourly timestamps, must include is_spike column).
    target: defaults to config.TARGET_COLUMN (read at call time); the active
            target is written back to config so feature helpers agree.
    """
    target = target or config.TARGET_COLUMN
    config.TARGET_COLUMN = target
    t_start_all = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    target_raw = df_features[target].astype(float)
    spike = df_features["is_spike"].astype(int)

    # Row positions of each calendar day, computed once (not per eval day)
    day_positions = df_features.groupby(df_features.index.normalize()).indices
    days_all = pd.DatetimeIndex(sorted(day_positions))

    def day_rows(d: pd.Timestamp) -> pd.DataFrame:
        pos = day_positions.get(d)
        return df_features.iloc[pos] if pos is not None else df_features.iloc[:0]

    eval_days = days_all[days_all >= pd.Timestamp(eval_start)]
    if max_eval_days is not None:
        eval_days = eval_days[:max_eval_days]

    # LSTM training window: the last TRAIN_DAYS days before each forecast day.
    train_days = config.TRAIN_DAYS

    arima_state = ArimaDayAhead()
    lstm_state: lstm_model.LstmSequenceModel | None = None
    lstm_weights: list[np.ndarray] | None = None
    last_lstm_refit_day: pd.Timestamp | None = None
    last_order_day: pd.Timestamp | None = None

    rows: list[dict] = []
    timings: list[dict] = []

    # Track the previous day's actuals for the ARIMA daily update
    prev_day_actual: np.ndarray | None = None

    for i, day in enumerate(eval_days):
        t_day = time.perf_counter()
        fday_feat = day_rows(day)
        prev_day_feat = day_rows(day - pd.Timedelta(days=1))
        day_actual = target_raw[fday_feat.index].values
        if len(day_actual) != config.PERIODS_PER_DAY:
            print(f"  [skip] {day.date()} — incomplete day ({len(day_actual)} periods)")
            continue

        # ---- Training window (everything strictly before `day`) ------------- #
        train_end = day - pd.Timedelta(minutes=30)  # last known period = end of prev day
        train_start = train_end - pd.Timedelta(days=train_days) + pd.Timedelta(minutes=30)
        train_series = target_raw.loc[(target_raw.index >= train_start) & (target_raw.index <= train_end)]
        if len(train_series) < config.PERIODS_PER_DAY * 300:
            print(f"  [warn] {day.date()} — short training window ({len(train_series)})")

        # ---- ARIMA: order selection (quarterly, or once per ORDER_SELECT_ONCE) #
        if last_order_day is None or (
            not config.ORDER_SELECT_ONCE
            and (day - last_order_day).days >= config.ARIMA_ORDER_REFIT_DAYS
        ):
            t0 = time.perf_counter()
            arima_state.order = arima_model.select_order_scaled(train_series.values)
            timings.append({"day": day, "component": "arima_order_select", "seconds": time.perf_counter() - t0})
            last_order_day = day

        # ---- ARIMA: weekly refit or daily state update ---------------------- #
        if arima_state.model is None or (
            arima_state.last_refit_day is not None
            and (day - arima_state.last_refit_day).days >= config.ARIMA_REFIT_DAYS
        ):
            dt = arima_state.refit(train_series.values, arima_state.order)
            timings.append({"day": day, "component": "arima_refit", "seconds": dt})
            arima_state.last_refit_day = day
        else:
            # daily update with the previously observed day
            if prev_day_actual is not None:
                arima_state.update(prev_day_actual)
            timings.append({"day": day, "component": "arima_update", "seconds": 0.0})

        # ---- ARIMA day-ahead prediction -------------------------------------- #
        arima_pred_raw = arima_state.predict()
        timings.append({"day": day, "component": "arima_predict", "seconds": 0.0})

        # ---- Seasonal naive baseline (same period yesterday) ----------------- #
        naive_pred = np.full(config.PERIODS_PER_DAY, np.nan)
        if len(prev_day_feat) == config.PERIODS_PER_DAY:
            naive_pred = target_raw[prev_day_feat.index].values

        # ---- LSTM residual model: monthly warm-start refit ------------------- #
        if lstm_state is None or (
            last_lstm_refit_day is not None
            and (day - last_lstm_refit_day).days >= config.LSTM_REFIT_DAYS
        ):
            t0 = time.perf_counter()
            lstm_state = fit_residual_lstm(
                df_features, target_raw, arima_state.order, train_end,
                weights_path=out_dir / f"lstm_{target}.weights.h5",
                initial_weights=lstm_weights,
            )
            lstm_weights = lstm_state.model.get_weights()
            last_lstm_refit_day = day
            timings.append({"day": day, "component": "lstm_refit", "seconds": time.perf_counter() - t0})
            print(f"  [lstm] refit at {day.date()} — {lstm_state.train_time_seconds:.0f}s")

        # ---- LSTM residual prediction for the forecast day ------------------- #
        X_seq_f, X_ctx_f = preprocessing.build_forecast_sequence(prev_day_feat, fday_feat)
        lstm_resid_raw = lstm_model.predict_sequence(lstm_state, X_seq_f, X_ctx_f)

        hybrid_pred_raw = hybrid_base(naive_pred, arima_pred_raw) + lstm_resid_raw

        # ---- Record ---------------------------------------------------------- #
        ts = fday_feat.index
        rows.extend(
            {
                "timestamp": t,
                "actual": a,
                "naive_pred": n,
                "arima_pred": ar,
                "lstm_resid": lr,
                "hybrid_pred": h,
                "is_spike": s,
            }
            for t, a, n, ar, lr, h, s in zip(
                ts, day_actual, naive_pred, arima_pred_raw, lstm_resid_raw,
                hybrid_pred_raw, spike[ts].values,
            )
        )
        timings.append({"day": day, "component": "total_day", "seconds": time.perf_counter() - t_day})

        prev_day_actual = day_actual

        if verbose and (i % 10 == 0 or i == len(eval_days) - 1):
            rmse_ar = float(np.sqrt(np.mean((day_actual - arima_pred_raw) ** 2)))
            rmse_hy = float(np.sqrt(np.mean((day_actual - hybrid_pred_raw) ** 2)))
            print(
                f"  [{i + 1}/{len(eval_days)}] {day.date()}  "
                f"ARIMA RMSE {rmse_ar:6.2f} | Hybrid RMSE {rmse_hy:6.2f}"
            )

    preds = pd.DataFrame(rows).set_index("timestamp").sort_index()
    timings_df = pd.DataFrame(timings)

    preds.to_csv(out_dir / f"predictions_{target}.csv")
    timings_df.to_csv(out_dir / f"timings_{target}.csv")
    print(f"\n[walkforward] {target}: {len(preds)} periods over {len(eval_days)} days -> {out_dir}")
    print(f"[walkforward] total {time.perf_counter() - t_start_all:.1f}s")

    return WalkForwardResult(
        predictions=preds,
        timings=timings_df,
        eval_start=eval_days[0],
        eval_end=eval_days[-1],
        n_days=len(eval_days),
    )
