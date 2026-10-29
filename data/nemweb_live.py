"""
Intraday NEM data from NEMWeb "Current" reports, for a true day-ahead forecast.

AEMO's Aggregated Price & Demand file (update_live.py) only runs through the
previous midnight, so on day D the last complete day is D-1 and the "next day"
is D itself. To forecast D+1 while D is still in progress, this module
  1. downloads today's 5-min dispatch results (DispatchIS) for all regions,
  2. downloads the latest AEMO pre-dispatch run (PredispatchIS), which covers
     the rest of today and all of tomorrow, and
  3. writes a "nowcast" price/demand file: actuals through the latest dispatch
     interval, AEMO pre-dispatch for the rest of today.

All times are NEM time (AEST, UTC+10, no daylight saving).

Binning convention matches update_live.py: 5-min values are labelled by their
interval-ENDING SETTLEMENTDATE and resampled with "30min" mean, so bin L holds
intervals ending L..L+25. Pre-dispatch periods are labelled by their ending
DATETIME T (intervals ending T-25..T); they are mapped to bin T-30min, which is
off by one 5-min interval. That approximation only affects the filled-in part
of today.
"""

from __future__ import annotations

import csv
import io
import re
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DATA_DIR as RAW_DIR, REGION  # noqa: E402

BASE = "https://nemweb.com.au/Reports/Current/"
DISPATCH_DIR = BASE + "DispatchIS_Reports/"
PREDISPATCH_DIR = BASE + "PredispatchIS_Reports/"

CACHE_DIR = RAW_DIR / "intraday"
NEM_TZ = timezone(timedelta(hours=10))


def nem_now() -> datetime:
    return datetime.now(NEM_TZ).replace(tzinfo=None)


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 dsh-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _listing(url: str, prefix: str) -> list[str]:
    html = _get(url).decode("utf-8", "replace")
    return sorted(set(re.findall(rf"{prefix}_[0-9_]+\.zip", html, flags=re.I)))


def _tables(zip_bytes: bytes, wanted: set[str]) -> dict[str, pd.DataFrame]:
    """Parse an AEMO multi-table CSV (I = header row, D = data row)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        text = zf.read(zf.namelist()[0]).decode("utf-8", "replace")
    headers, rows = {}, {}
    for parts in csv.reader(text.splitlines()):
        if len(parts) < 4:
            continue
        key = f"{parts[1]}.{parts[2]}"
        if key not in wanted:
            continue
        if parts[0] == "I":
            headers[key] = parts[4:]
        elif parts[0] == "D":
            rows.setdefault(key, []).append(parts[4:])
    return {
        k: pd.DataFrame([r[: len(headers[k])] for r in v], columns=headers[k])
        for k, v in rows.items()
    }


# ----------------------------------------------------------------------------- #
# Dispatch (5-min actuals)
# ----------------------------------------------------------------------------- #
DISPATCH_TABLES = {"DISPATCH.PRICE", "DISPATCH.REGIONSUM", "DISPATCH.INTERCONNECTORRES"}


def _dispatch_file(name: str) -> dict[str, pd.DataFrame]:
    path = CACHE_DIR / name.replace(".zip", ".pkl")
    if path.exists():
        return pd.read_pickle(path)
    t = _tables(_get(DISPATCH_DIR + name), DISPATCH_TABLES)
    pd.to_pickle(t, path)
    return t


def fetch_dispatch_today(day: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    """5-min price, demand and interconnector flows for every region, for the
    intervals ending (day 00:05 .. now]. Downloads are cached per file."""
    day = pd.Timestamp(day or nem_now().date())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = day.strftime("%Y%m%d")
    names = [n for n in _listing(DISPATCH_DIR, "PUBLIC_DISPATCHIS")
             if n.upper().startswith(f"PUBLIC_DISPATCHIS_{stamp}")]
    names = [n for n in names if n.split("_")[2][8:12] != "0000"]   # 00:00 belongs to yesterday
    if not names:
        raise RuntimeError(f"No DispatchIS files on NEMWeb for {day.date()}")
    with ThreadPoolExecutor(max_workers=12) as ex:
        parts = list(ex.map(_dispatch_file, names))
    print(f"[nemweb] dispatch: {len(parts)} intervals for {day.date()}")

    def stack(key: str, cols: list[str]) -> pd.DataFrame:
        df = pd.concat([p[key] for p in parts if key in p], ignore_index=True)
        df = df[df["INTERVENTION"] == "0"]
        df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    price = stack("DISPATCH.PRICE", ["RRP"])
    regsum = stack("DISPATCH.REGIONSUM", ["TOTALDEMAND", "NETINTERCHANGE"])
    ic = stack("DISPATCH.INTERCONNECTORRES", ["MWFLOW", "EXPORTLIMIT", "IMPORTLIMIT"])
    return {
        "price": price.pivot_table(index="SETTLEMENTDATE", columns="REGIONID", values="RRP"),
        "demand": regsum.pivot_table(index="SETTLEMENTDATE", columns="REGIONID", values="TOTALDEMAND"),
        "interconnectors": ic[ic["SETTLEMENTDATE"] == ic["SETTLEMENTDATE"].max()][
            ["INTERCONNECTORID", "MWFLOW", "EXPORTLIMIT", "IMPORTLIMIT"]
        ].reset_index(drop=True),
    }


# ----------------------------------------------------------------------------- #
# Pre-dispatch (AEMO's own 30-min forecast, to the end of tomorrow)
# ----------------------------------------------------------------------------- #
def fetch_predispatch() -> tuple[pd.DataFrame, str]:
    """Latest pre-dispatch run: 30-min RRP and TOTALDEMAND per region, indexed
    by period-ENDING DATETIME. Returns (frame, run_label)."""
    name = _listing(PREDISPATCH_DIR, "PUBLIC_PREDISPATCHIS")[-1]
    t = _tables(_get(PREDISPATCH_DIR + name), {"PREDISPATCH.REGION_PRICES", "PREDISPATCH.REGION_SOLUTION"})
    rp, rs = t["PREDISPATCH.REGION_PRICES"], t["PREDISPATCH.REGION_SOLUTION"]
    rp, rs = rp[rp["INTERVENTION"] == "0"], rs[rs["INTERVENTION"] == "0"]
    out = pd.DataFrame({
        "DATETIME": pd.to_datetime(rp["DATETIME"], format="%Y/%m/%d %H:%M:%S"),
        "REGIONID": rp["REGIONID"],
        "price": pd.to_numeric(rp["RRP"], errors="coerce"),
    }).merge(
        pd.DataFrame({
            "DATETIME": pd.to_datetime(rs["DATETIME"], format="%Y/%m/%d %H:%M:%S"),
            "REGIONID": rs["REGIONID"],
            "demand": pd.to_numeric(rs["TOTALDEMAND"], errors="coerce"),
        }),
        on=["DATETIME", "REGIONID"], how="left",
    )
    run = name.split("_")[2]           # e.g. 202610091830
    print(f"[nemweb] pre-dispatch run {run}: {out['DATETIME'].min()} .. {out['DATETIME'].max()}")
    return out, run


def predispatch_bins(pd_df: pd.DataFrame, region: str = REGION) -> pd.DataFrame:
    """Pre-dispatch for one region on the project's 30-min bin labels."""
    r = pd_df[pd_df["REGIONID"] == region].set_index("DATETIME").sort_index()
    r.index = r.index - pd.Timedelta(minutes=30)
    r.index.name = "timestamp"
    return r[["price", "demand"]]


# ----------------------------------------------------------------------------- #
# Nowcast file
# ----------------------------------------------------------------------------- #
def build_nowcast(
    live_path: Path,
    dispatch: dict[str, pd.DataFrame],
    pd_bins: pd.DataFrame,
    out_path: Path,
    day: pd.Timestamp,
) -> pd.DataFrame:
    """Live 30-min file + today's dispatch actuals + pre-dispatch for the rest
    of today. Returns a per-bin 'source' frame for today (actual / predispatch)."""
    live = pd.read_csv(live_path, index_col=0, parse_dates=True)
    day = pd.Timestamp(day).normalize()

    # 5-min NSW series: the interval ending at today 00:00 comes from the
    # monthly AEMO file (it is the last row there), the rest from dispatch.
    month_raw = RAW_DIR / "live" / f"PRICE_AND_DEMAND_{day:%Y%m}_{REGION}.csv"
    m = pd.read_csv(month_raw)
    m = m[m["REGION"] == REGION]
    m.index = pd.to_datetime(m["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    five = pd.DataFrame({"price": m["RRP"], "demand": m["TOTALDEMAND"]}).astype(float)
    disp = pd.DataFrame({"price": dispatch["price"][REGION], "demand": dispatch["demand"][REGION]})
    five = pd.concat([five, disp[~disp.index.isin(five.index)]]).sort_index()
    five = five[five.index >= day]
    actual30 = five.resample("30min").mean()

    bins = pd.date_range(day, periods=48, freq="30min")
    today = actual30.reindex(bins)
    source = pd.Series("actual", index=bins)
    missing = today["price"].isna()
    today.loc[missing] = pd_bins.reindex(bins[missing]).values
    source[missing] = "predispatch"
    if today.isna().any().any():
        raise RuntimeError("Pre-dispatch does not cover the rest of today")

    # Like the AEMO daily file, end on the next midnight bin: the loader needs a
    # row after 23:30 to interpolate hourly temperature onto it.
    boundary = pd_bins.reindex([day + pd.Timedelta(days=1)])
    full = pd.concat([live[live.index < day], today, boundary]).sort_index()
    full.index.name = "timestamp"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path)
    n_act = int((source == "actual").sum())
    print(f"[nemweb] nowcast: today {n_act}/48 bins actual, {48 - n_act} from pre-dispatch -> {out_path}")
    return pd.DataFrame({"source": source, "price": today["price"], "demand": today["demand"]})
