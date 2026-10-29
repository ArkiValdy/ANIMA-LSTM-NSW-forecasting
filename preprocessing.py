"""
Feature engineering, spike/peak flagging, and day-ahead sequence building.

Day-ahead availability rule (Option A): a feature value at period `p` of day
D+1 may only use information known at the end of day D. This means:

  - Lag features: shifts of >= 1 day (all >= 24h) — always known.
  - Rolling statistics: computed with a one-day shift (shift(48)), so a
    window's last value is the previous day, not the current row.
  - Calendar features of the forecast day are known (they are the day being
    predicted).
  - Weather (hdd/cdd) of the forecast day uses the ACTUAL temperature as a
    "perfect forecast" proxy (standard practice in load forecasting; stated
    in the Method section).

The LSTM consumes, for each forecast day D+1:
  - a sequence of the previous day's (D) 48 per-period feature vectors, and
  - a broadcast context vector of D+1's calendar + weather features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


# ----------------------------------------------------------------------------- #
# Containers
# ----------------------------------------------------------------------------- #
@dataclass
class Splits:
    """Static chronological split (kept only as a quick smoke test)."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    target_scaler: StandardScaler
    y_test_raw: np.ndarray
    spike_test: np.ndarray
    idx_test: pd.DatetimeIndex


# ----------------------------------------------------------------------------- #
# Spike / peak detection (target-aware)
# ----------------------------------------------------------------------------- #
def flag_spike_periods(series: pd.Series) -> pd.Series:
    """Flag extreme periods for the ACTIVE target.

    - price target: sliding-window mean + k*std OR hard AUD/MWh threshold.
    - demand target: top-quantile demand periods (peak demand).
    """
    if config.TARGET_COLUMN == "demand":
        thr = series.quantile(config.PEAK_DEMAND_QUANTILE)
        return (series >= thr).astype(int)

    rolling = series.rolling(
        window=config.OUTLIER_WINDOW, min_periods=config.OUTLIER_WINDOW // 4
    )
    upper = rolling.mean() + config.OUTLIER_K * rolling.std()
    is_spike = (series > upper) | (series > config.SPIKE_THRESHOLD)
    return is_spike.fillna(False).astype(int)


# ----------------------------------------------------------------------------- #
# Feature engineering
# ----------------------------------------------------------------------------- #
def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df.index.dayofweek.astype(int)
    df["month"] = df.index.month.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    au_holidays = holidays.country_holidays("AU", subdiv="NSW")
    df["is_holiday"] = (
        df.index.normalize().to_series().isin(au_holidays).astype(int).values
    )
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    period_of_day = np.arange(len(df)) % config.PERIODS_PER_DAY
    df["hour_sin"] = np.sin(2 * np.pi * period_of_day / config.PERIODS_PER_DAY)
    df["hour_cos"] = np.cos(2 * np.pi * period_of_day / config.PERIODS_PER_DAY)
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for d in config.PRICE_LAG_DAYS:
        df[f"price_lag_{d}d"] = df["price"].shift(d * config.PERIODS_PER_DAY)
    for d in config.DEMAND_LAG_DAYS:
        df[f"demand_lag_{d}d"] = df["demand"].shift(d * config.PERIODS_PER_DAY)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling stats with a ONE-DAY shift so they are known at day-ahead time."""
    for col in ("price", "demand"):
        shifted = df[col].shift(config.PERIODS_PER_DAY)  # value known at end of prev day
        for d in config.ROLLING_WINDOWS_DAYS:
            win = d * config.PERIODS_PER_DAY
            rolling = shifted.rolling(window=win, min_periods=win // 4)
            df[f"{col}_roll_mean_{d}d"] = rolling.mean()
            df[f"{col}_roll_std_{d}d"] = rolling.std()
    return df


def _add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    base = config.BASE_TEMPERATURE
    df["hdd"] = (base - df["temperature"]).clip(lower=0)
    df["cdd"] = (df["temperature"] - base).clip(lower=0)
    return df


def _add_forecast_day_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-period forecast-day demand (actual as perfect-forecast proxy)."""
    df["fday_demand_delta_gw"] = (df["demand"] - df["demand_lag_1d"]) / 1000.0
    df["fday_demand_z"] = (
        (df["demand"] - df["demand_roll_mean_7d"]) / df["demand_roll_std_7d"]
    )
    return df


def build_features(df: pd.DataFrame, forecast_day=None) -> pd.DataFrame:
    """
    Build the full feature matrix on the merged price/demand/temperature frame.
    Adds an `is_spike` column computed on the active target (price spikes or
    peak demand), preserved through the rest of the pipeline.

    forecast_day: in forecast mode, the (future) day whose rows carry only
    temperature; they are kept although their target is NaN.
    """
    df = df.copy()

    cols = ["price", "demand", "temperature"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    df["is_spike"] = flag_spike_periods(df[config.TARGET_COLUMN])

    df = _add_temperature_features(df)
    df = _add_calendar_features(df)
    df = _add_time_features(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_forecast_day_demand_features(df)

    n_before = len(df)
    required = config.SEQ_FEATURE_SUFFIXES + period_context_columns() + [config.TARGET_COLUMN]
    keep = df[required].notna().all(axis=1)
    if forecast_day is not None:
        # The day being forecast has no target yet; it only needs its context
        in_fday = df.index.normalize() == pd.Timestamp(forecast_day).normalize()
        keep |= in_fday & df[context_feature_columns()].notna().all(axis=1)
    df = df.loc[keep]
    print(
        f"[preprocessing] Dropped {n_before - len(df)} warm-up rows; "
        f"{len(df)} remain."
    )
    return df


def seq_feature_columns() -> list:
    """Per-period sequence feature columns for the active target."""
    return list(config.SEQ_FEATURE_SUFFIXES)


def context_feature_columns() -> list:
    return list(config.CONTEXT_FEATURES)


def period_context_columns() -> list:
    """Per-period forecast-day context columns (not day-averaged)."""
    if config.TARGET_COLUMN == "price" and config.PRICE_USE_FDAY_DEMAND:
        return list(config.FDAY_DEMAND_FEATURES)
    return []


# ----------------------------------------------------------------------------- #
# Day-ahead sequence building
# ----------------------------------------------------------------------------- #
def _day_groups(idx: pd.DatetimeIndex) -> np.ndarray:
    """Array of length len(idx) with a stable integer per calendar day."""
    days = idx.normalize()
    _, day_ids = np.unique(days.values, return_inverse=True)
    return day_ids


def _period_encoding_2d(n: int = config.PERIODS_PER_DAY) -> np.ndarray:
    """(n, 2) sin/cos encoding of period-of-day index (used in the context)."""
    idx = np.arange(n)
    phase = 2.0 * np.pi * idx / n
    return np.stack([np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32)


def _ctx_day_matrix(
    ctx_mean: np.ndarray, n_periods: int, per_period: np.ndarray | None = None
) -> np.ndarray:
    """Broadcast a day-mean context vector across periods, with period encoding.

    Returns (n_periods, n_ctx + 2 [+ n_pp]): each row = day context + (sin, cos)
    of that period's index [+ that period's own per-period context values].
    """
    period_enc = _period_encoding_2d(n_periods)                      # (n, 2)
    ctx_b = np.tile(ctx_mean[None, :], (n_periods, 1))               # (n, C)
    parts = [ctx_b, period_enc]
    if per_period is not None and per_period.shape[1] > 0:
        parts.append(per_period)                                     # (n, P)
    return np.concatenate(parts, axis=-1).astype(np.float32)


def build_sequences(
    df: pd.DataFrame,
    target: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """
    Build day-ahead LSTM training data from the feature frame.

    For each day D+1 with a full previous day D available:
      X_seq[d] = (48, n_seq) feature vectors of day D
      X_ctx[d] = (48, n_ctx+2[+n_pp]) forecast-day context per period, where
                 each row = day-mean (calendar + weather) + period-of-day
                 encoding [+ per-period forecast-day demand, price target]
      y[d]     = (48,) target values of day D+1 (raw scale)

    Returns (X_seq, X_ctx, y, seq_cols, ctx_cols).
    """
    target_col = target or config.TARGET_COLUMN

    seq_cols = seq_feature_columns()
    ctx_cols = context_feature_columns()
    pp_cols = period_context_columns()

    day_ids = _day_groups(df.index)
    periods_per_day = config.PERIODS_PER_DAY

    days_with_target = (
        df.groupby(day_ids)[target_col].count().values == periods_per_day
    )
    full_day_ids = np.where(days_with_target)[0]

    # A sample exists for day D+1 iff both day D and day D+1 are full days
    # and D+1's features (lags) are computable — guaranteed by dropna above.
    usable = full_day_ids[np.isin(full_day_ids + 1, full_day_ids)]

    X_seq = np.empty((len(usable), periods_per_day, len(seq_cols)), dtype=np.float32)
    X_ctx = np.empty(
        (len(usable), periods_per_day, len(ctx_cols) + 2 + len(pp_cols)), dtype=np.float32
    )
    y = np.empty((len(usable), periods_per_day), dtype=np.float32)

    for i, d in enumerate(usable):
        prev_mask = day_ids == d
        nxt_mask = day_ids == (d + 1)
        X_seq[i] = df.loc[prev_mask, seq_cols].values.astype(np.float32)
        # Day-mean context (calendar + weather) + per-period encoding
        ctx_mean = df.loc[nxt_mask, ctx_cols].mean(axis=0).values.astype(np.float32)
        per_period = df.loc[nxt_mask, pp_cols].values.astype(np.float32)
        X_ctx[i] = _ctx_day_matrix(ctx_mean, periods_per_day, per_period)
        y[i] = df.loc[nxt_mask, target_col].values.astype(np.float32)

    return X_seq, X_ctx, y, seq_cols, ctx_cols


def build_forecast_sequence(
    df_prev_day: pd.DataFrame, df_forecast_day: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the LSTM input for ONE forecast day from the two feature frames:
      - df_prev_day: rows of the previous day (48 periods), used as the sequence
      - df_forecast_day: rows of the forecast day, used for the context
    Returns (X_seq (1,48,n_seq), X_ctx (1,48,n_ctx+2)) in float32.
    """
    seq_cols = seq_feature_columns()
    ctx_cols = context_feature_columns()
    pp_cols = period_context_columns()

    seq = df_prev_day[seq_cols].values.astype(np.float32)
    ctx_mean = df_forecast_day[ctx_cols].mean(axis=0).values.astype(np.float32)

    assert seq.shape[0] == config.PERIODS_PER_DAY, (
        f"Previous day must have {config.PERIODS_PER_DAY} periods, got {seq.shape[0]}"
    )
    per_period = df_forecast_day[pp_cols].values.astype(np.float32)
    ctx = _ctx_day_matrix(ctx_mean, config.PERIODS_PER_DAY, per_period)
    return seq.reshape(1, config.PERIODS_PER_DAY, -1), ctx.reshape(1, config.PERIODS_PER_DAY, -1)


# ----------------------------------------------------------------------------- #
# Static split (smoke test only; headline results come from walk_forward.py)
# ----------------------------------------------------------------------------- #
def _chronological_split(n: int) -> Tuple[slice, slice, slice]:
    n_train = int(n * 0.55)
    n_val = int(n * 0.15)
    return (
        slice(0, n_train),
        slice(n_train, n_train + n_val),
        slice(n_train + n_val, n),
    )


def prepare(df_features: pd.DataFrame) -> Splits:
    """
    Chronologically split into train/val/test, fit scalers on the training
    portion only. Kept for a quick smoke test; NOT the headline evaluation.
    """
    feature_cols = config.SEQ_FEATURE_SUFFIXES
    target_col = config.TARGET_COLUMN

    X = df_features[feature_cols].values.astype(np.float32)
    y = df_features[target_col].values.astype(np.float32).reshape(-1, 1)
    spike = df_features["is_spike"].values.astype(int)
    idx = df_features.index

    s_tr, s_va, s_te = _chronological_split(len(df_features))

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train = feature_scaler.fit_transform(X[s_tr])
    X_val = feature_scaler.transform(X[s_va])
    X_test = feature_scaler.transform(X[s_te])

    y_train = target_scaler.fit_transform(y[s_tr]).ravel()
    y_val = target_scaler.transform(y[s_va]).ravel()
    y_test = target_scaler.transform(y[s_te]).ravel()

    # StandardScaler preserves the float32 input dtype
    return Splits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        target_scaler=target_scaler,
        y_test_raw=y[s_te].ravel(),
        spike_test=spike[s_te],
        idx_test=idx[s_te],
    )
