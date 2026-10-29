"""
Live data updater: extends the study dataset up to the latest available day.

The MMSDM monthly archive (download_aemo.py) lags by 1-2 months, so recent
price/demand comes from AEMO's Aggregated Price & Demand files, which are
updated through the current day. Temperature comes from the Open-Meteo
archive (a few days' lag). Safe to re-run: the current month is always
re-downloaded.

The original raw files are left untouched (plan Sec 1); new consolidated
files are written next to them:
  data/raw/live/PRICE_AND_DEMAND_{yyyymm}_NSW1.csv   raw AEMO downloads
  data/raw/price_demand_30min_live.csv               historical + live
  data/raw/temperature_hourly_live.csv               historical + live

Usage:
  python data/update_live.py               # live from 2026-07 to today
  python main.py --mode walkforward --target price --eval_start 2026-07-01 \
      --price_demand data/raw/price_demand_30min_live.csv \
      --temperature data/raw/temperature_hourly_live.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_temperature  # noqa: E402
from download_aemo import RAW_DIR, REGION, iter_months  # noqa: E402
from nemweb_live import _get  # noqa: E402

PD_URL = "https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{yyyymm}_{region}.csv"

LIVE_DIR = RAW_DIR / "live"


def load_price_demand_month(year: int, month: int, today: date) -> pd.DataFrame:
    """Download (or reuse) one month of 5-min price/demand and resample to 30 min."""
    yyyymm = f"{year:04d}{month:02d}"
    path = LIVE_DIR / f"PRICE_AND_DEMAND_{yyyymm}_{REGION}.csv"
    is_current = (year, month) == (today.year, today.month)
    if path.exists() and not is_current:
        print(f"  [skip] exists: {path.name}")
    else:
        print(f"  [get ] {yyyymm}")
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(PD_URL.format(yyyymm=yyyymm, region=REGION), timeout=120))

    df = pd.read_csv(path)
    df = df[df["REGION"] == REGION]
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    df = df.set_index("SETTLEMENTDATE").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    out = pd.DataFrame(
        {
            "price": pd.to_numeric(df["RRP"], errors="coerce"),
            "demand": pd.to_numeric(df["TOTALDEMAND"], errors="coerce"),
        }
    )
    # Same 5-min -> 30-min aggregation as download_aemo.load_month
    return out.resample("30min").mean()


def _months(start: str, today: date):
    """(year, month) from start ("YYYY-MM") through today's month, inclusive."""
    return iter_months(start, f"{today.year}-{today.month:02d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-07", help="first live month, YYYY-MM")
    args = ap.parse_args()
    today = date.today()
    live_start = pd.Timestamp(f"{args.start}-01")

    # ---- price / demand ---------------------------------------------------- #
    hist = pd.read_csv(RAW_DIR / "price_demand_30min.csv", index_col=0, parse_dates=True)
    live = pd.concat(load_price_demand_month(y, m, today) for y, m in _months(args.start, today))
    full = pd.concat([hist[hist.index < live_start], live[live.index >= live_start]])
    full = full.dropna(how="all").sort_index()
    full.index.name = "timestamp"
    out = RAW_DIR / "price_demand_30min_live.csv"
    full.to_csv(out)
    print(f"[live] price/demand: {len(full)} rows ({full.index.min()} .. {full.index.max()}) -> {out.name}")

    # ---- temperature -------------------------------------------------------- #
    t_hist = pd.read_csv(RAW_DIR / "temperature_hourly.csv", index_col=0, parse_dates=True)
    t_live = download_temperature.fetch(
        live_start.strftime("%Y-%m-%d"), (today - timedelta(days=1)).isoformat()
    ).dropna()
    t_full = pd.concat([t_hist[t_hist.index < live_start], t_live]).sort_index()
    t_full = t_full[~t_full.index.duplicated(keep="last")]
    t_full.index.name = "timestamp"
    t_out = RAW_DIR / "temperature_hourly_live.csv"
    t_full.to_csv(t_out)
    print(f"[live] temperature:  {len(t_full)} rows ({t_full.index.min()} .. {t_full.index.max()}) -> {t_out.name}")


if __name__ == "__main__":
    sys.exit(main())
