"""
AEMO historical data downloader (MMSDM SQL-loader archives).

Downloads monthly per-table zips for the MMSDM data archive and parses the
AEMO "C/I/D" row format into clean CSVs:

  - C rows  -> control rows (skipped)
  - I rows  -> table header (column names)
  - D rows  -> data rows

Outputs:
  data/raw/zips/            raw zips kept untouched (provenance, plan Sec 1)
  data/raw/price_demand_30min.csv   consolidated half-hourly price+demand
                                    (NSW1 only), ready for the pipeline

Tables used:
  DISPATCHPRICE     5-min regional RRP          (actual)
  DISPATCHREGIONSUM 5-min regional TOTALDEMAND  (actual)
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DATA_DIR as RAW_DIR, REGION  # noqa: E402

# The MMSDM SQL-loader archive is the data source.
BASE = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
# (table, value column) for the actual 5-min price and demand series.
TABLES = (("DISPATCHPRICE", "RRP"), ("DISPATCHREGIONSUM", "TOTALDEMAND"))

ZIP_DIR = RAW_DIR / "zips"


def table_url(year: int, month: int, table: str) -> str:
    """URL of one month's archived zip for an MMSDM table."""
    yyyymm = f"{year:04d}{month:02d}"
    return (
        f"{BASE}/{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA/"
        f"PUBLIC_ARCHIVE%23{table}%23FILE01%23{yyyymm}010000.zip"
    )


def download(url: str, dest: Path, timeout: int = 120) -> None:
    """Download url to dest in chunks via a .part file; skip if dest already
    exists and is non-empty."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] exists: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "dsh-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.rename(dest)
    print(f"  [done] {dest.stat().st_size / 1e6:.1f} MB")


def parse_aemo_csv(raw: bytes) -> pd.DataFrame:
    """Parse the C/I/D MMS row format into a DataFrame: C (control) rows are
    skipped, the I row gives the column names, D rows are the data."""
    text = raw.decode("utf-8", errors="replace")
    header: list[str] | None = None
    rows: list[list[str]] = []
    for line in csv.reader(io.StringIO(text)):
        if not line:
            continue
        kind = line[0]
        if kind == "I":
            header = [c.strip() for c in line]
        elif kind == "D" and header is not None:
            rows.append(line)
    if header is None or not rows:
        raise ValueError("No I (header) or D (data) rows found in AEMO CSV")
    return pd.DataFrame(rows, columns=header)


def iter_months(start: str, end: str) -> Iterator[tuple[int, int]]:
    """Yield (year, month) from start to end inclusive, both "YYYY-MM"."""
    y, m = (int(x) for x in start.split("-"))
    end_ym = tuple(int(x) for x in end.split("-"))
    while (y, m) <= end_ym:
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def load_month(year: int, month: int) -> pd.DataFrame:
    """Download + parse one month of price and demand, merged on timestamp."""
    series = []
    for table, col in TABLES:
        zip_path = ZIP_DIR / f"{table}_{year:04d}{month:02d}.zip"
        download(table_url(year, month, table), zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            df = parse_aemo_csv(zf.read(zf.namelist()[0]))
        df["SETTLEMENTDATE"] = pd.to_datetime(
            df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S", errors="coerce"
        )
        # Keep only the target region, no-intervention rows
        df = df[(df["REGIONID"] == REGION) & (df["INTERVENTION"] == "0")]
        series.append(df[["SETTLEMENTDATE", col]].copy())

    price, demand = series
    merged = price.merge(demand, on="SETTLEMENTDATE", how="outer").sort_values(
        "SETTLEMENTDATE"
    )
    merged["price"] = pd.to_numeric(merged["RRP"], errors="coerce")
    merged["demand"] = pd.to_numeric(merged["TOTALDEMAND"], errors="coerce")
    merged = merged[["SETTLEMENTDATE", "price", "demand"]].dropna()
    merged = merged.set_index("SETTLEMENTDATE")
    merged = merged[~merged.index.duplicated(keep="first")]

    # Aggregate 5-min -> 30-min to match the pipeline period
    return merged.resample("30min").mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01", help="YYYY-MM inclusive")
    ap.add_argument("--end", default="2026-06", help="YYYY-MM inclusive")
    args = ap.parse_args()

    parts: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    for y, m in iter_months(args.start, args.end):
        print(f"[aemo] {y:04d}-{m:02d}")
        df = load_month(y, m)
        print(f"  -> {len(df)} half-hourly rows  ({df.index.min()} .. {df.index.max()})")
        parts.append(df)

    full = pd.concat(parts)
    full = full[~full.index.duplicated(keep="first")].sort_index()
    full.index.name = "timestamp"
    out = RAW_DIR / "price_demand_30min.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(out)
    print(f"\n[aemo] Wrote {len(full)} rows to {out}")
    print(f"[aemo] Range: {full.index.min()} .. {full.index.max()}")
    print(f"[aemo] Elapsed: {time.perf_counter() - t0:.1f}s")
    print(full.describe().round(2).to_string())


if __name__ == "__main__":
    sys.exit(main())
