"""
Data loading utilities for AEMO half-hourly price/demand and BOM temperature data.

The expected layouts are flexible — the loader looks for common AEMO column
names (SETTLEMENTDATE, RRP, TOTALDEMAND, REGION) and a temperature file with a
datetime column and a temperature column. Anything reasonable is accepted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

import config


PathLike = Union[str, Path]


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column from `candidates` that exists in `df`."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _to_30min(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate timestamps (keep first) and mean-resample to 30 minutes."""
    return df[~df.index.duplicated(keep="first")].resample("30min").mean()


def _parse_datetime_column(df: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """Parse the first matching candidate column as datetime and set as index."""
    col = _find_column(df, candidates)
    if col is None:
        raise ValueError(
            f"None of the expected datetime columns {candidates} were found. "
            f"Available columns: {list(df.columns)}"
        )
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    df = df.set_index(col).sort_index()
    df.index.name = "timestamp"
    return df


# ----------------------------------------------------------------------------- #
# Public loaders
# ----------------------------------------------------------------------------- #
def load_price_demand(path: PathLike, region: str = config.REGION) -> pd.DataFrame:
    """
    Load an AEMO price-and-demand CSV and return a DataFrame indexed by
    half-hourly timestamps with columns ['price', 'demand'].

    The source file is published at 5-minute resolution; we mean-aggregate
    to 30-minute periods to match the rest of the pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Price/demand file not found: {path}")

    df = pd.read_csv(path)
    df = _parse_datetime_column(df, ["SETTLEMENTDATE", "datetime", "timestamp"])

    region_col = _find_column(df, ["REGION"])
    if region_col is not None and region is not None:
        df = df[df[region_col].astype(str).str.upper() == region.upper()]

    price_col = _find_column(df, ["RRP", "price"])
    demand_col = _find_column(df, ["TOTALDEMAND", "demand", "total_demand"])
    if price_col is None or demand_col is None:
        raise ValueError(
            "Could not locate price (RRP) or demand (TOTALDEMAND) columns."
        )

    out = pd.DataFrame(
        {
            "price": pd.to_numeric(df[price_col], errors="coerce"),
            "demand": pd.to_numeric(df[demand_col], errors="coerce"),
        },
        index=df.index,
    )

    return _to_30min(out)


def load_temperature(path: PathLike) -> pd.DataFrame:
    """
    Load a temperature CSV and return a DataFrame indexed by 30-minute
    timestamps with column 'temperature' (Celsius).

    Two source formats are supported:

    1. BoM daily product (e.g. IDCJAC0010_*): separate Year/Month/Day columns
       with a 'Maximum temperature (Degree C)' value. The daily value is
       broadcast across every 30-min slot of that day.
    2. A pre-built file with a single datetime column and a temperature column,
       which is mean-resampled to 30 minutes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Temperature file not found: {path}")

    df = pd.read_csv(path)

    # ---- Format 1: BoM daily product (Year/Month/Day columns) -------------- #
    y, m, d = (_find_column(df, [c]) for c in ("Year", "Month", "Day"))
    if y and m and d:
        temp_col = _find_column(
            df,
            [
                "Maximum temperature (Degree C)",
                "Minimum temperature (Degree C)",
                "temperature",
                "temp",
            ],
        )
        if temp_col is None:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            for c in (y, m, d):
                if c in numeric_cols:
                    numeric_cols.remove(c)
            if not numeric_cols:
                raise ValueError("No temperature column found in BoM CSV.")
            temp_col = numeric_cols[0]

        dates = pd.to_datetime(
            dict(
                year=pd.to_numeric(df[y], errors="coerce"),
                month=pd.to_numeric(df[m], errors="coerce"),
                day=pd.to_numeric(df[d], errors="coerce"),
            ),
            errors="coerce",
        )
        daily = (
            pd.DataFrame(
                {"temperature": pd.to_numeric(df[temp_col], errors="coerce").values},
                index=pd.DatetimeIndex(dates.values),
            )
            .dropna()
            .sort_index()
        )
        daily = daily[~daily.index.duplicated(keep="first")]

        # Broadcast each daily value across that day's 30-min slots
        slot_index = pd.date_range(
            start=daily.index.min(),
            end=daily.index.max() + pd.Timedelta(days=1) - pd.Timedelta(minutes=30),
            freq="30min",
        )
        out = daily.reindex(slot_index, method="ffill")
        out.index.name = "timestamp"
        return out

    # ---- Format 2: generic datetime + temperature column ------------------- #
    df = _parse_datetime_column(df, ["SETTLEMENTDATE", "datetime", "timestamp", "date", "time"])

    temp_col = _find_column(df, ["temperature", "temp", "AIR_TEMP", "air_temperature"])
    if temp_col is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            raise ValueError("No temperature column found in temperature CSV.")
        temp_col = numeric_cols[0]

    out = pd.DataFrame(
        {"temperature": pd.to_numeric(df[temp_col], errors="coerce")},
        index=df.index,
    )
    return _to_30min(out)


def _handle_missing(df: pd.DataFrame, max_gap_hours: float = 4.0) -> pd.DataFrame:
    """
    Linear-interpolate gaps shorter than `max_gap_hours`, drop rows still NaN.
    """
    max_gap_periods = int(max_gap_hours * (60 / 30))  # half-hour periods
    interpolated = df.interpolate(method="linear", limit=max_gap_periods, limit_area="inside")
    n_dropped = interpolated.isna().any(axis=1).sum()
    if n_dropped:
        print(f"[data_loader] Dropping {n_dropped} rows with gaps > {max_gap_hours}h")
    return interpolated.dropna()


def read_live(path: PathLike = config.LIVE_PRICE_DEMAND_CSV) -> pd.DataFrame:
    """Read a timestamp-indexed price/demand CSV as-is (no resampling),
    keeping the first row of any duplicated timestamp."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[~df.index.duplicated(keep="first")]


def _load_consolidated(price_demand_path: PathLike) -> pd.DataFrame:
    """
    Load the consolidated 30-min price/demand CSV produced by
    `data/download_aemo.py` (index = timestamp, columns = price, demand).
    """
    df = read_live(price_demand_path)
    df.index.name = "timestamp"
    return df.resample("30min").mean()


def load_dataset(
    price_demand_path: PathLike = config.PRICE_DEMAND_CSV,
    temperature_path: Optional[PathLike] = config.TEMPERATURE_CSV,
    region: str = config.REGION,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and merge price, demand, and temperature into a single half-hourly
    DataFrame ready for feature engineering.

    Supports two price/demand layouts:
      1. Consolidated 30-min CSV (timestamp, price, demand) — the default.
      2. Raw AEMO monthly/daily files (SETTLEMENTDATE, RRP, TOTALDEMAND ...),
         which are mean-aggregated to 30 minutes.

    Temperature is loaded from an hourly/daily file and resampled to 30
    minutes. If the temperature file is missing, a synthetic seasonal
    temperature is generated so that the pipeline still runs end-to-end
    (clearly flagged in the output).
    """
    path = Path(price_demand_path)
    if path.exists():
        sample = pd.read_csv(path, nrows=5)
        col_lower = [c.lower() for c in sample.columns]
        if {"price", "demand"}.issubset(col_lower) and "timestamp" in col_lower:
            pd_df = _load_consolidated(path)
        else:
            pd_df = load_price_demand(path, region=region)
    else:
        raise FileNotFoundError(f"Price/demand file not found: {path}")

    if temperature_path is not None and Path(temperature_path).exists():
        temp_df = load_temperature(temperature_path)
        df = pd_df.join(temp_df, how="left")
    else:
        if verbose:
            print(
                "[data_loader] WARNING: temperature file not found; "
                "generating a synthetic seasonal temperature signal so the "
                "pipeline can run. Replace with real BOM data for production."
            )
        df = pd_df.copy()
        # Synthetic NSW-ish temperature: yearly + daily seasonality + noise
        rng = np.random.default_rng(config.RANDOM_SEED)
        day_of_year = df.index.dayofyear.values
        hour_of_day = df.index.hour.values + df.index.minute.values / 60.0
        yearly = 18.0 + 8.0 * np.cos(2 * np.pi * (day_of_year - 30) / 365.25)
        daily = 4.0 * np.sin(2 * np.pi * (hour_of_day - 6) / 24.0)
        df["temperature"] = yearly + daily + rng.normal(0, 1.0, size=len(df))

    df = _handle_missing(df, max_gap_hours=4.0)

    if verbose:
        summarise(df)

    return df


def summarise(df: pd.DataFrame) -> None:
    """Print a concise summary of the merged dataset."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Shape           : {df.shape}")
    print(f"Date range      : {df.index.min()}  ->  {df.index.max()}")
    print(f"Frequency       : {pd.infer_freq(df.index[:200])}")
    print(f"Missing values  :")
    for col in df.columns:
        n_missing = df[col].isna().sum()
        print(f"   {col:<14} {n_missing}")
    print("\nBasic statistics:")
    print(df.describe().round(2).to_string())
    print("=" * 60 + "\n")
